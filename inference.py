import os
from data.base_dataset import get_params, get_transform
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import save_image, tensor2im
from util.visualizer import save_images
from util import html
from PIL import Image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.num_threads = 0             # Limiting threads for testing
    opt.batch_size = 1              # Batch size for testing
    opt.serial_batches = True       # Disable shuffling for consistent results
    opt.no_flip = True              # Disable image flipping for testing
    opt.display_id = -1             # No visdom display during testing
    
    # Create the dataset based on specified options
    dataset = create_dataset(opt)
    
    # Create the model based on specified options
    model = create_model(opt)
    model.setup(opt)            # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    A_path = "datasets/mri2ct/testA/mri149.jpg"

    input_nc = 3
    A = Image.open(A_path).convert('RGB')
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))
    
    A = A_transform(A).unsqueeze(0)
    data = {"A":A,"A_paths":[A_path]}
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory

    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()

    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals() 

    
    img_path = ["assets/real_img.png","assets/fake_img.png"]

    for i,(label, im_data) in enumerate(visuals.items()):
        im = tensor2im(im_data)
        save_path = img_path[i]
        save_image(im, save_path, aspect_ratio=1.0)

    save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
