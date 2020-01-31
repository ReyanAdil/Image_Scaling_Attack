import ScalingAttack as sa
from PIL import Image

if __name__ == "__main__":
    source_image = Image
    target_image = Image
    try:
        # Source Image
        source_image = Image.open("sheep.jpg")

        # Target Image
        target_image = Image.open("wolf.jpg")
    except IOError:
        print("File Not Found")
        exit(-1)

    sa.implement_attack(source_image, target_image)