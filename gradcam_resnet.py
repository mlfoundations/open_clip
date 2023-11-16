from src.gradcam.gradcam import grad_cam

model_name = "RN50"
pretrain_tag = "openai"
image_name = "test.jpg"
caption_text = "a cat"

grad_cam(
    model_name=model_name,
    pretrain_tag=pretrain_tag,
    image_name=image_name,
    caption_text=caption_text,
)
