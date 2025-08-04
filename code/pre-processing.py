import cv2
import os


def standardize_images(input_folder, output_folder, CLAHE=False, image_size=None, binary=False):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Iterate through all files in a folder
    for filename in os.listdir(input_folder):
        # Check if the file is in a common image format
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read image
            image = cv2.imread(input_path)

            # Color images only
            if image is not None and len(image.shape) == 3:
                # Image resizing using bilinear interpolation

                if CLAHE:
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    # Split HSV image into three channels
                    h, s, v = cv2.split(hsv_image)

                    # Apply CLAHE to the V channel
                    v_clahe = clahe.apply(v)

                    # Merge channels back to HSV image
                    hsv_clahe = cv2.merge((h, s, v_clahe))

                    # Converting an image back to BGR color space
                    image = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
                if image_size:
                    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
                if binary:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Binarisation, pixel value > 0 set to 255 (white), otherwise 0 (black)
                    _, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

                # 保存调整大小后的图像
                cv2.imwrite(output_path, image)
                print(f"Processed and saved: {filename}")
            else:
                print(f"Cannot load or is not a color image: {filename}")


if __name__ == "__main__":
    input_folder = 'A'
    output_folder = 'B'
    standardize_images(input_folder, output_folder, CLAHE=True, image_size=(300, 200), binary=True)
