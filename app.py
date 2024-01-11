import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

device = 'cpu'

sam_checkpoint = 'model_checkpoint/sam_vit_h_4b8939.pth'

model_type = 'vit_h'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

sam.to(device)

predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-inpainting',
        torch_dtype=torch.float16,
        )

pipe = pipe.to('cuda')


def inpaint(image, mask, prompt):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    image = image.resize([512,512])
    mask = mask.resize([512,512])

    output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            ).images[0]
    return output

def generate_mask(image, evt: gr.SelectData):
    selected_pixels.append(evt.index)
    predictor.set_image(image)
    input_points = np.array(selected_pixels)
    input_labels = np.ones(input_points.shape[0])
    mask, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
            )
    mask = np.logical_not(mask)
    mask = Image.fromarray(mask[0, :, :])
    return mask

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(label='Input')
        mask_image = gr.Image(label='Mask, black=mask=pixels do not require to edit')
        output_image = gr.Image(label='Output')
    with gr.Blocks():

        prompt_text = gr.Textbox(lines=1, label='Prompt')
    with gr.Row():
        submit = gr.Button('Submit')
    #print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB') 
    
    input_image.select(generate_mask, [input_image], [mask_image])
    
    #print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB') 
    submit.click(
            inpaint,
            inputs=[input_image, mask_image, prompt_text],
            outputs=[output_image]
            )

if __name__ == '__main__':
    demo.launch()
