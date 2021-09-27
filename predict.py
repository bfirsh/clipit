import sys
import os
from types import SimpleNamespace
from pathlib import Path
import tempfile
from PIL import ImageFile, Image, PngImagePlugin
import torch
from torch import nn, optim
from torch_optimizer import DiffGrad, AdamP, RAdam
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import cog
from CLIP import clip
from vqgan import VqganDrawer
from clipit import (SimpleNamespace, setup_parser, palette_from_string, MakeCutouts, random_noise_image,
                    old_random_noise_image, real_glob, parse_prompt, Prompt, resize_image, re_average_z)


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @cog.input("prompts", type=str, default='sunset in the city', help="Text prompts")
    @cog.input("quality", type=str, options=["draft", "normal", "better", "best"], default="normal", help="quality")
    @cog.input("display_every", type=int, default=20, options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               help="Display image iterations. For reference, the total number of iterations is determined by the "
                    "quality chosen above: draft=200, normal=300, better and best=500")
    @cog.input("aspect", type=str, options=["widescreen", "square"], default="widescreen",
               help="widescreen or square aspect")
    def predict(self, prompts='sunset in the city', quality='normal', display_every=20, aspect="widescreen"):

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        clipit = Clipit()
        clipit.add_settings(prompts=prompts, aspect=aspect, display_every=display_every,
                            quality=quality, output=str(out_path))
        settings = clipit.apply_settings()
        clipit.do_init(settings, self.device)
        clipit.do_run(settings, self.device, out_path)

        return out_path


class Clipit:
    def __init__(self):
        self.global_clipit_settings = {}
        self.global_padding_mode = 'reflection'
        self.global_aspect_width = 1
        self.global_spot_file = None
        self.z_orig = None
        self.z_targets = None
        self.z_labels = None
        self.opts = None
        self.drawer = None
        self.perceptors = {}
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
        self.cutoutsTable = {}
        self.cutoutSizeTable = {}
        self.init_image_tensor = None
        self.target_image_tensor = None
        self.pmsTable = {}
        self.pmsImageTable = {}
        self.spotPmsTable = {}
        self.spotOffPmsTable = {}
        self.gside_X = None
        self.gside_Y = None
        self.overlay_image_rgba = None
        self.cur_iteration = 0
        self.cur_anim_index = None
        self.anim_output_files = []
        self.anim_cur_zs = []
        self.anim_next_zs = []
        self.QUALITY_TO_CLIP_MODELS_TABLE = {
            'draft': 'ViT-B/32',
            'normal': 'ViT-B/32,ViT-B/16',
            'better': 'RN50,ViT-B/32,ViT-B/16',
            'best': 'RN50x4,ViT-B/32,ViT-B/16'
        }
        self.QUALITY_TO_ITERATIONS_TABLE = {
            'draft': 200,
            'normal': 350,
            'better': 500,
            'best': 500
        }
        self.QUALITY_TO_SCALE_TABLE = {
            'draft': 1,
            'normal': 2,
            'better': 3,
            'best': 4
        }
        self.QUALITY_TO_NUM_CUTS_TABLE = {
            'draft': 40,
            'normal': 40,
            'better': 40,
            'best': 40
        }
        self.SIZE_TO_SCALE_TABLE = {
            'small': 1,
            'medium': 2,
            'large': 4
        }
        self.ASPECT_TO_SIZE_TABLE = {
            'square': [150, 150],
            'widescreen': [200, 112]
        }

    def add_settings(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                self.global_clipit_settings.pop(k, None)
            else:
                self.global_clipit_settings[k] = v

    def apply_settings(self):
        settings_dict = None
        vq_parser = setup_parser()

        if len(self.global_clipit_settings) > 0:
            # check for any bogus entries in the settings
            dests = [d.dest for d in vq_parser._actions]
            for k in self.global_clipit_settings:
                if k not in dests:
                    raise ValueError(f"Requested setting not found, aborting: {k}={self.global_clipit_settings[k]}")

            settings_dict = SimpleNamespace(**self.global_clipit_settings)

        settings = self.process_args(vq_parser, settings_dict)
        return settings

    def process_args(self, vq_parser, namespace=None):
        if namespace is None:
            # command line: use ARGV to get args
            args = vq_parser.parse_args()
        else:
            # notebook, ignore ARGV and use dictionary instead
            args = vq_parser.parse_args(args=[], namespace=namespace)

        if args.cudnn_determinism:
            torch.backends.cudnn.deterministic = True

        if args.quality not in self.QUALITY_TO_CLIP_MODELS_TABLE:
            print("Quality setting not understood, aborting -> ", args.quality)
            exit(1)

        if args.clip_models is None:
            args.clip_models = self.QUALITY_TO_CLIP_MODELS_TABLE[args.quality]
        if args.iterations is None:
            args.iterations = self.QUALITY_TO_ITERATIONS_TABLE[args.quality]
        if args.num_cuts is None:
            args.num_cuts = self.QUALITY_TO_NUM_CUTS_TABLE[args.quality]
        if args.ezsize is None and args.scale is None:
            args.scale = self.QUALITY_TO_SCALE_TABLE[args.quality]

        if args.size is not None:
            self.global_aspect_width = args.size[0] / args.size[1]
        elif args.aspect == "widescreen":
            self.global_aspect_width = 16 / 9
        else:
            self.global_aspect_width = 1

        # determine size if not set
        if args.size is None:
            size_scale = args.scale
            if size_scale is None:
                if args.ezsize in self.SIZE_TO_SCALE_TABLE:
                    size_scale = self.SIZE_TO_SCALE_TABLE[args.ezsize]
                else:
                    print("EZ Size not understood, aborting -> ", args.ezsize)
                    exit(1)
            if args.aspect in self.ASPECT_TO_SIZE_TABLE:
                base_size = self.ASPECT_TO_SIZE_TABLE[args.aspect]
                base_width = int(size_scale * base_size[0])
                base_height = int(size_scale * base_size[1])
                args.size = [base_width, base_height]
            else:
                print("aspect not understood, aborting -> ", args.aspect)
                exit(1)

        if args.init_noise.lower() == "none":
            args.init_noise = None

        # Split text prompts using the pipe character
        if args.prompts:
            args.prompts = [phrase.strip() for phrase in args.prompts.split("|")]

        # Split text prompts using the pipe character
        if args.spot_prompts:
            args.spot_prompts = [phrase.strip() for phrase in args.spot_prompts.split("|")]

        # Split text prompts using the pipe character
        if args.spot_prompts_off:
            args.spot_prompts_off = [phrase.strip() for phrase in args.spot_prompts_off.split("|")]

        # Split text labels using the pipe character
        if args.labels:
            args.labels = [phrase.strip() for phrase in args.labels.split("|")]

        # Split target images using the pipe character
        if args.image_prompts:
            args.image_prompts = args.image_prompts.split("|")
            args.image_prompts = [image.strip() for image in args.image_prompts]

        if args.target_palette is not None:
            args.target_palette = palette_from_string(args.target_palette)

        if args.overlay_every is not None and args.overlay_every <= 0:
            args.overlay_every = None

        clip_models = args.clip_models.split(",")
        args.clip_models = [model.strip() for model in clip_models]

        # Make video steps directory
        if args.make_video:
            if not os.path.exists('steps'):
                os.mkdir('steps')

        self.global_spot_file = args.spot_file

        return args

    def do_init(self, args, device):
        # args.use_pixeldraw = False, use_clipdraw=False
        self.drawer = VqganDrawer(args.vqgan_model)
        self.drawer.load_model(args.vqgan_config, args.vqgan_checkpoint, device)
        num_resolutions = self.drawer.get_num_resolutions()

        jit = True if float(torch.__version__[:3]) < 1.8 else False
        f = 2 ** (num_resolutions - 1)

        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f

        self.gside_X = sideX
        self.gside_Y = sideY

        for clip_model in args.clip_models:
            perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
            self.perceptors[clip_model] = perceptor

            cut_size = perceptor.visual.input_resolution
            self.cutoutSizeTable[clip_model] = cut_size
            if cut_size not in self.cutoutsTable:
                make_cutouts = MakeCutouts(cut_size, args.num_cuts, cut_pow=args.cut_pow)
                self.cutoutsTable[cut_size] = make_cutouts

        # Image initialisation
        if args.init_image or args.init_noise:
            if args.init_noise == 'pixels':
                img = random_noise_image(args.size[0], args.size[1])
            elif args.init_noise == 'gradient':
                img = random_gradient_image(args.size[0], args.size[1])
            elif args.init_noise == 'snow':
                img = old_random_noise_image(args.size[0], args.size[1])
            else:
                img = Image.new(mode="RGB", size=(args.size[0], args.size[1]), color=(255, 255, 255))
            starting_image = img.convert('RGB')
            starting_image = starting_image.resize((sideX, sideY), Image.LANCZOS)

            if args.init_image:
                # now we might overlay an init image (init_image also can be recycled as overlay)
                if 'http' in args.init_image:
                    init_image = Image.open(urlopen(args.init_image))
                else:
                    init_image = Image.open(args.init_image)
                # this version is needed potentially for the loss function
                init_image_rgb = init_image.convert('RGB')
                init_image_rgb = init_image_rgb.resize((sideX, sideY), Image.LANCZOS)
                self.init_image_tensor = TF.to_tensor(init_image_rgb)
                self.init_image_tensor = self.init_image_tensor.to(device).unsqueeze(0)

                # this version gets overlaid on the background (noise)
                init_image_rgba = init_image.convert('RGBA')
                init_image_rgba = init_image_rgba.resize((sideX, sideY), Image.LANCZOS)
                top_image = init_image_rgba.copy()
                if args.init_image_alpha and args.init_image_alpha >= 0:
                    top_image.putalpha(args.init_image_alpha)
                starting_image.paste(top_image, (0, 0), top_image)

            starting_image.save("starting_image.png")
            starting_tensor = TF.to_tensor(starting_image)
            init_tensor = starting_tensor.to(device).unsqueeze(0) * 2 - 1
            self.drawer.init_from_tensor(init_tensor)

        else:
            # untested
            self.drawer.rand_init(toksX, toksY)

        if args.overlay_every:
            if args.overlay_image:
                if 'http' in args.overlay_image:
                    overlay_image = Image.open(urlopen(args.overlay_image))
                else:
                    overlay_image = Image.open(args.overlay_image)
                self.overlay_image_rgba = overlay_image.convert('RGBA')
                self.overlay_image_rgba = self.overlay_image_rgba.resize((sideX, sideY), Image.LANCZOS)
            else:
                self.overlay_image_rgba = init_image_rgba
            if args.overlay_alpha:
                self.overlay_image_rgba.putalpha(args.overlay_alpha)
            self.overlay_image_rgba.save('overlay_image.png')

        if args.target_images is not None:
            self.z_targets = []
            filelist = real_glob(args.target_images)
            for target_image in filelist:
                target_image = Image.open(target_image)
                target_image_rgb = target_image.convert('RGB')
                target_image_rgb = target_image_rgb.resize((sideX, sideY), Image.LANCZOS)
                target_image_tensor_local = TF.to_tensor(target_image_rgb)
                self.target_image_tensor = target_image_tensor_local.to(device).unsqueeze(0) * 2 - 1
                z_target = self.drawer.get_z_from_tensor(target_image_tensor)
                self.z_targets.append(z_target)

        if args.image_labels is not None:
            self.z_labels = []
            filelist = real_glob(args.image_labels)
            cur_labels = []
            for image_label in filelist:
                image_label = Image.open(image_label)
                image_label_rgb = image_label.convert('RGB')
                image_label_rgb = image_label_rgb.resize((sideX, sideY), Image.LANCZOS)
                image_label_rgb_tensor = TF.to_tensor(image_label_rgb)
                image_label_rgb_tensor = image_label_rgb_tensor.to(device).unsqueeze(0) * 2 - 1
                z_label = self.drawer.get_z_from_tensor(image_label_rgb_tensor)
                cur_labels.append(z_label)
            image_embeddings = torch.stack(cur_labels)
            print("Processing labels: ", image_embeddings.shape)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings.mean(dim=0)
            image_embeddings /= image_embeddings.norm()
            self.z_labels.append(image_embeddings.unsqueeze(0))

        self.z_orig = self.drawer.get_z_copy()

        for clip_model in args.clip_models:
            self.pmsTable[clip_model] = []
            self.pmsImageTable[clip_model] = []
            self.spotPmsTable[clip_model] = []
            self.spotOffPmsTable[clip_model] = []

        # CLIP tokenize/encode
        # NR: Weights / blending
        for prompt in args.prompts:
            for clip_model in args.clip_models:
                pMs = self.pmsTable[clip_model]
                perceptor = self.perceptors[clip_model]
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.spot_prompts:
            for clip_model in args.clip_models:
                pMs = self.spotPmsTable[clip_model]
                perceptor = self.perceptors[clip_model]
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.spot_prompts_off:
            for clip_model in args.clip_models:
                pMs = self.spotOffPmsTable[clip_model]
                perceptor = self.perceptors[clip_model]
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

        for label in args.labels:
            for clip_model in args.clip_models:
                pMs = self.pmsTable[clip_model]
                perceptor = self.perceptors[clip_model]
                txt, weight, stop = parse_prompt(label)
                texts = [template.format(txt) for template in imagenet_templates]  # format with class
                print(f"Tokenizing all of {texts}")
                texts = clip.tokenize(texts).to(device)  # tokenize
                class_embeddings = perceptor.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                pMs.append(Prompt(class_embedding.unsqueeze(0), weight, stop).to(device))

        for clip_model in args.clip_models:
            pImages = self.pmsImageTable[clip_model]
            for prompt in args.image_prompts:
                path, weight, stop = parse_prompt(prompt)
                img = Image.open(path)
                pil_image = img.convert('RGB')
                img = resize_image(pil_image, (sideX, sideY))
                pImages.append(TF.to_tensor(img).unsqueeze(0).to(device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))

        self.opts = self.drawer.get_opts()
        if self.opts is None:
            # legacy

            # Set the optimiser
            z = self.drawer.get_z()
            if args.optimiser == "Adam":
                opt = optim.Adam([z], lr=args.learning_rate)  # LR=0.1
            elif args.optimiser == "AdamW":
                opt = optim.AdamW([z], lr=args.learning_rate)  # LR=0.2
            elif args.optimiser == "Adagrad":
                opt = optim.Adagrad([z], lr=args.learning_rate)  # LR=0.5+
            elif args.optimiser == "Adamax":
                opt = optim.Adamax([z], lr=args.learning_rate)  # LR=0.5+?
            elif args.optimiser == "DiffGrad":
                opt = DiffGrad([z], lr=args.learning_rate)  # LR=2+?
            elif args.optimiser == "AdamP":
                opt = AdamP([z], lr=args.learning_rate)  # LR=2+?
            elif args.optimiser == "RAdam":
                opt = RAdam([z], lr=args.learning_rate)  # LR=2+?

            self.opts = [opt]

        # Output for the user
        print('Using device:', device)
        print('Optimising using:', args.optimiser)

        if args.prompts:
            print('Using text prompts:', args.prompts)
        if args.spot_prompts:
            print('Using spot prompts:', args.spot_prompts)
        if args.spot_prompts_off:
            print('Using spot off prompts:', args.spot_prompts_off)
        if args.image_prompts:
            print('Using image prompts:', args.image_prompts)
        if args.init_image:
            print('Using initial image:', args.init_image)
        if args.noise_prompt_weights:
            print('Noise prompt weights:', args.noise_prompt_weights)

        if args.seed is None:
            seed = torch.seed()
        else:
            seed = args.seed
        torch.manual_seed(seed)
        print('Using seed:', seed)

    def do_run(self, args, device, out_path):

        if args.animation_dir is not None:
            if not os.path.exists(args.animation_dir):
                os.mkdir(args.animation_dir)
            filelist = real_glob(args.target_images)
            num_anim_frames = len(filelist)
            for target_image in filelist:
                basename = os.path.basename(target_image)
                target_output = os.path.join(args.animation_dir, basename)
                self.anim_output_files.append(target_output)
            for i in range(num_anim_frames):
                cur_z = self.drawer.get_z_copy()
                self.anim_cur_zs.append(cur_z)
                self.anim_next_zs.append(None)

            step_iteration = 0

            with tqdm() as pbar:
                while True:
                    cur_images = []
                    for i in range(num_anim_frames):
                        # do merge frames here from cur->next when we are ready to be fancy
                        self.cur_anim_index = i
                        # self.anim_cur_zs[cur_anim_index] = self.anim_next_zs[cur_anim_index]
                        self.cur_iteration = step_iteration
                        self.drawer.set_z(anim_cur_zs[cur_anim_index])
                        for j in range(args.save_every):
                            self.do_train(args, self.cur_iteration, out_path)
                            self.cur_iteration += 1
                            pbar.update()

                        cur_images.append(drawer.to_image())
                    step_iteration = step_iteration + args.save_every
                    if step_iteration >= args.iterations:
                        break
                    # compute the next round of cur_zs here from all the next_zs
                    for i in range(num_anim_frames):
                        prev_i = (i + num_anim_frames - 1) % num_anim_frames
                        base_image = cur_images[i].copy()
                        prev_image = cur_images[prev_i].copy().convert('RGBA')
                        prev_image.putalpha(args.animation_alpha)
                        base_image.paste(prev_image, (0, 0), prev_image)
                        # base_image.save(f"overlaid_{i:02d}.png")
                        self.drawer.reapply_from_tensor(TF.to_tensor(base_image).to(device).unsqueeze(0) * 2 - 1)
                        anim_cur_zs[i] = self.drawer.get_z_copy()
        else:
            try:
                with tqdm() as pbar:
                    while True:
                        try:
                            self.do_train(args, self.cur_iteration, out_path)
                            if self.cur_iteration == args.iterations:
                                break
                            self.cur_iteration += 1
                            pbar.update()
                        except RuntimeError as e:
                            print("Oops: runtime error: ", e)
                            print("Try reducing --num-cuts to save memory")
                            raise e
            except KeyboardInterrupt:
                pass

    def do_train(self, args, cur_it, out_path):
        print(f'do_train {cur_it}')
        for opt in self.opts:
            # opt.zero_grad(set_to_none=True)
            opt.zero_grad()

        for i in range(args.batches):
            lossAll = self.ascend_txt(args)

            if i == 0 and cur_it % args.save_every == 0:
                yield self.checkin(args, cur_it, lossAll, out_path)

            loss = sum(lossAll)
            loss.backward()

        for opt in self.opts:
            opt.step()

        if args.overlay_every and cur_it != 0 and (cur_it % (args.overlay_every + args.overlay_offset)) == 0:
            re_average_z(args)

        self.drawer.clip_z()

    def ascend_txt(self, args):
        print('ascend_txt')
        out = self.drawer.synth(self.cur_iteration)

        result = []

        if self.cur_iteration % 2 == 0:
            self.global_padding_mode = 'reflection'
        else:
            self.global_padding_mode = 'border'

        cur_cutouts = {}
        cur_spot_cutouts = {}
        cur_spot_off_cutouts = {}
        for cutoutSize in self.cutoutsTable:
            make_cutouts = self.cutoutsTable[cutoutSize]
            cur_cutouts[cutoutSize] = make_cutouts(out)

        if args.spot_prompts:
            for cutoutSize in self.cutoutsTable:
                cur_spot_cutouts[cutoutSize] = make_cutouts(out, spot=1)

        if args.spot_prompts_off:
            for cutoutSize in self.cutoutsTable:
                cur_spot_off_cutouts[cutoutSize] = make_cutouts(out, spot=0)

        for clip_model in args.clip_models:
            perceptor = self.perceptors[clip_model]
            cutoutSize = self.cutoutSizeTable[clip_model]
            transient_pMs = []

            if args.spot_prompts:
                iii_s = perceptor.encode_image(self.normalize(cur_spot_cutouts[cutoutSize])).float()
                spotPms = self.spotPmsTable[clip_model]
                for prompt in spotPms:
                    result.append(prompt(iii_s))

            if args.spot_prompts_off:
                iii_so = perceptor.encode_image(self.normalize(cur_spot_off_cutouts[cutoutSize])).float()
                spotOffPms = self.spotOffPmsTable[clip_model]
                for prompt in spotOffPms:
                    result.append(prompt(iii_so))

            pMs = self.pmsTable[clip_model]
            iii = perceptor.encode_image(self.normalize(cur_cutouts[cutoutSize])).float()
            for prompt in pMs:
                result.append(prompt(iii))

            # If there are image prompts we make cutouts for those each time
            # so that they line up with the current cutouts from augmentation
            make_cutouts = self.cutoutsTable[cutoutSize]
            pImages = self.pmsImageTable[clip_model]
            for timg in pImages:
                # note: this caches and reuses the transforms - a bit of a hack but it works

                if args.image_prompt_shuffle:
                    # print("Disabling cached transforms")
                    make_cutouts.transforms = None

                # print("Building throwaway image prompts")
                # new way builds throwaway Prompts
                batch = make_cutouts(timg)
                embed = perceptor.encode_image(normalize(batch)).float()
                if args.image_prompt_weight is not None:
                    transient_pMs.append(Prompt(embed, args.image_prompt_weight).to(device))
                else:
                    transient_pMs.append(Prompt(embed).to(device))

            for prompt in transient_pMs:
                result.append(prompt(iii))

        if args.enforce_palette_annealing and args.target_palette:
            target_palette = torch.FloatTensor(args.target_palette).requires_grad_(False).to(device)
            _pixels = cur_cutouts[cutoutSize].permute(0, 2, 3, 1).reshape(-1, 3)
            palette_dists = torch.cdist(target_palette, _pixels, p=2)
            best_guesses = palette_dists.argmin(axis=0)
            diffs = _pixels - target_palette[best_guesses]
            palette_loss = torch.mean(torch.norm(diffs, 2, dim=1)) * cur_cutouts[cutoutSize].shape[0]
            result.append(palette_loss * self.cur_iteration / args.enforce_palette_annealing)

        if args.enforce_smoothness and args.enforce_smoothness_type:
            _pixels = cur_cutouts[cutoutSize].permute(0, 2, 3, 1).reshape(-1, cur_cutouts[cutoutSize].shape[2], 3)
            gyr, gxr = torch.gradient(_pixels[:, :, 0])
            gyg, gxg = torch.gradient(_pixels[:, :, 1])
            gyb, gxb = torch.gradient(_pixels[:, :, 2])
            sharpness = torch.sqrt(gyr ** 2 + gxr ** 2 + gyg ** 2 + gxg ** 2 + gyb ** 2 + gxb ** 2)
            if args.enforce_smoothness_type == 'clipped':
                sharpness = torch.clamp(sharpness, max=0.5)
            elif args.enforce_smoothness_type == 'log':
                sharpness = torch.log(torch.ones_like(sharpness) + sharpness)
            sharpness = torch.mean(sharpness)

            result.append(sharpness * self.cur_iteration / args.enforce_smoothness)

        if args.enforce_saturation:
            # based on the old "percepted colourfulness" heuristic from Hasler and Süsstrunk’s 2003 paper
            # https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images
            _pixels = cur_cutouts[cutoutSize].permute(0, 2, 3, 1).reshape(-1, 3)
            rg = _pixels[:, 0] - _pixels[:, 1]
            yb = 0.5 * (_pixels[:, 0] + _pixels[:, 1]) - _pixels[:, 2]
            rg_std, rg_mean = torch.std_mean(rg)
            yb_std, yb_mean = torch.std_mean(yb)
            std_rggb = torch.sqrt(rg_std ** 2 + yb_std ** 2)
            mean_rggb = torch.sqrt(rg_mean ** 2 + yb_mean ** 2)
            colorfullness = std_rggb + .3 * mean_rggb

            result.append(-colorfullness * self.cur_iteration / args.enforce_saturation)

        for cutoutSize in self.cutoutsTable:
            # clear the transform "cache"
            make_cutouts = self.cutoutsTable[cutoutSize]
            make_cutouts.transforms = None

        # main init_weight uses spherical loss
        if args.target_images is not None and args.target_image_weight > 0:
            if self.cur_anim_index is None:
                cur_z_targets = self.z_targets
            else:
                cur_z_targets = [z_targets[cur_anim_index]]
            for z_target in cur_z_targets:
                f2 = z_target.reshape(1, -1)
                cur_loss = spherical_dist_loss(f, f2) * args.target_image_weight
                result.append(cur_loss)

        if args.target_weight_pix:
            if self.target_image_tensor is None:
                print("OOPS TIT is 0")
            else:
                cur_loss = F.l1_loss(out, self.target_image_tensor) * args.target_weight_pix
                result.append(cur_loss)

        if args.image_labels is not None:
            for z_label in self.z_labels:
                f = self.drawer.get_z().reshape(1, -1)
                f2 = z_label.reshape(1, -1)
                cur_loss = spherical_dist_loss(f, f2) * args.image_label_weight
                result.append(cur_loss)

        # main init_weight uses spherical loss
        if args.init_weight:
            f = self.drawer.get_z().reshape(1, -1)
            f2 = self.z_orig.reshape(1, -1)
            cur_loss = spherical_dist_loss(f, f2) * args.init_weight
            result.append(cur_loss)

        # these three init_weight variants offer mse_loss, mse_loss in pixel space, and cos loss
        if args.init_weight_dist:
            cur_loss = F.mse_loss(z, self.z_orig) * args.init_weight_dist / 2
            result.append(cur_loss)

        if args.init_weight_pix:
            if self.init_image_tensor is None:
                print("OOPS IIT is 0")
            else:
                cur_loss = F.l1_loss(out, self.init_image_tensor) * args.init_weight_pix / 2
                result.append(cur_loss)

        if args.init_weight_cos:
            f = self.drawer.get_z().reshape(1, -1)
            f2 = self.z_orig.reshape(1, -1)
            y = torch.ones_like(f[0])
            cur_loss = F.cosine_embedding_loss(f, f2, y) * args.init_weight_cos
            result.append(cur_loss)

        if args.make_video:
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
            img = np.transpose(img, (1, 2, 0))
            imageio.imwrite(f'./steps/frame_{cur_iteration:04d}.png', np.array(img))

        return result

    @torch.no_grad()
    def checkin(self, args, iter, losses, out_path):
        print('calling checkin')
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        writestr = f'iter: {iter}, loss: {sum(losses).item():g}, losses: {losses_str}'
        # tqdm.write(writestr)
        sys.stderr.write(writestr)
        info = PngImagePlugin.PngInfo()
        info.add_text('comment', f'{args.prompts}')
        img = self.drawer.to_image()
        if self.cur_anim_index is None:
            outfile = args.output
        else:
            outfile = self.anim_output_files[cur_anim_index]
        img.save(outfile, pnginfo=info)

        if iter % args.display_every == 0:
            if self.cur_anim_index is None or iter == 0:
                # display.display(display.Image(outfile))
                # Image(outfile).save(str(out_path))
                img.save(outfile, pnginfo=info)
        return out_path

