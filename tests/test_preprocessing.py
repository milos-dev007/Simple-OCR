from PIL import Image, ImageDraw

from ocr.preprocessing import preprocess_pil_image


def test_preprocessing_outputs_expected_shape():
    image = Image.new("L", (120, 40), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 80, 30), fill=0)

    tensor = preprocess_pil_image(image)

    assert tuple(tensor.shape) == (1, 32, 256)
