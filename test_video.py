import cv2
import os
import networks
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from torchvision import transforms, datasets
import PIL.Image as pil
from utils import download_model_if_doesnt_exist


def play_demo(video_path, model_name):
    # play video
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        ok, frame = vid.read()
        if ok == False:
            break
        cv2.imshow("result", frame)
        cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


def video_demo(video_path, model_name):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    print("-> Loading model from ", model_path)
    print("-> encoder_path = ", encoder_path)
    print("-> depth_decoder_path = ", depth_decoder_path)

    # LOADING PRETRAINED MODEL
    print("-> Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("-> Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # play video
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        ok, frame = vid.read()
        if ok == False:
            break
        with torch.no_grad():
            # Load image and preprocess
            input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # print(input_image)
            original_width, original_height = input_image.size
            #print("-> input image size ", input_image.size)
            input_image = input_image.resize(
                (feed_width, feed_height), pil.LANCZOS)
            #print("-> resize to ", input_image.size)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            # print(input_image)

            # PREDICTION
            input_image = input_image.to(device)
            # print(input_image)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(
                vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[
                              :, :, :3] * 255).astype(np.uint8)

            cv2.imshow('img', colormapped_im)
            cv2.waitKey(10)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        #cv2.imshow("result", frame)
        # cv2.waitKey(30)
        #key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #    break
    cv2.destroyAllWindows()


def cv2_to_PIL():
    image = cv2.imread('lena.png')
    image = pil.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.show()


def PIL_to_cv2():
    image = pil.open('lena.png')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.imshow('lena', image)
    cv2.waitKey()


def image_demo(image_path, model_name):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    print("-> Loading model from ", model_path)
    print("-> encoder_path = ", encoder_path)
    print("-> depth_decoder_path = ", depth_decoder_path)

    # LOADING PRETRAINED MODEL
    print("-> Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("-> Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    print("-> Predicting test image : ", image_path)

    with torch.no_grad():
        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        print(input_image)
        original_width, original_height = input_image.size
        print("-> input image size ", input_image.size)
        input_image = input_image.resize(
            (feed_width, feed_height), pil.LANCZOS)
        print("-> resize to ", input_image.size)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        print(input_image)

        # PREDICTION
        input_image = input_image.to(device)
        print(input_image)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(
            vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[
                          :, :, :3] * 255).astype(np.uint8)

        cv2.imshow('img', colormapped_im)
        cv2.waitKey(0)


if __name__ == "__main__":
    # camera_demo()
    model_name = 'mono+stereo_640x192'
    imagefile = 'C:/20.jpg'
    videofile = 'C:/001.mp4'
    image_demo(imagefile, model_name)
    #video_demo(videofile, model_name)
    #play_demo(videofile, model_name)
