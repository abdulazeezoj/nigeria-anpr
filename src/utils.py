import cv2
import os
import uuid
import json
import glob
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from ultralytics import YOLO


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

ALL_DATA_PATH = os.path.join(DATA_PATH, "all")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

MODEL_WEIGHT_PATH = os.path.join(MODEL_PATH, "weight")
CLASS_NAMES_PATH = os.path.join(DATA_PATH, "class.names")

CONSOLE = Console()


# Function to extract frames from a video
def extract_frame(input_loc: str, output_loc: str, verbose: bool = False):
    vidcap = cv2.VideoCapture(input_loc)

    # Retrieve the total number of frames in the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    # Print video information to the console
    table = Table(
        title="Video Information",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )
    table.add_column("Information", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Total Frames", str(total_frames))
    table.add_row("FPS", str(vid_fps))
    table.add_row("Duration", f"{total_frames / vid_fps} seconds")
    CONSOLE.print(table, new_line_start=True)

    try:
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
    except OSError:
        CONSOLE.print("[red][ ERROR ] : Unable to create directory[/red]")

    # Extract frames from the video showing a progress bar
    for i in track(range(total_frames), description="Extracting frames..."):
        success, image = vidcap.read()

        if not success:
            break

        name = os.path.join(output_loc, f"{uuid.uuid4()}.jpg")

        if verbose:
            CONSOLE.print(f"[cyan][ INFO ] : Saving {name}[/cyan]")

        try:
            cv2.imwrite(name, image)
        except Exception:
            CONSOLE.print("[red][ ERROR ] : Unable to save frame[/red]")

    vidcap.release()
    cv2.destroyAllWindows()


# Function to convert labelme annotations to YOLO format
def parse_data():
    input_loc = os.path.join(DATA_PATH, "raw")
    output_image_loc = os.path.join(DATA_PATH, "all/images")
    output_label_loc = os.path.join(DATA_PATH, "all/labels")

    classes = Path(CLASS_NAMES_PATH).read_text().splitlines()

    try:
        if not os.path.exists(output_image_loc):
            os.makedirs(output_image_loc)

        if not os.path.exists(output_label_loc):
            os.makedirs(output_label_loc)
    except OSError:
        CONSOLE.print("[red][ ERROR ] : Unable to create directory[/red]")

    images = [img for img in glob.glob(f"{input_loc}/*.jpg")]
    total = len(images)
    converted = 0
    copied = 0

    for image in track(images, description="Converting labels..."):
        filename = os.path.splitext(os.path.basename(image))[0]

        src = os.path.join(input_loc, filename)
        img_dst = os.path.join(output_image_loc, f"{filename}.jpg")
        label_dst = os.path.join(output_label_loc, f"{filename}.txt")

        # Copy image to output directory
        try:
            shutil.copy(f"{src}.jpg", img_dst)

            copied += 1
        except Exception as exp:
            CONSOLE.print(f"[red][ ERROR ] : {exp}")

        # Convert labelme annotations to YOLO format
        try:
            with open(f"{src}.json", "r") as f:
                data = json.load(f)

            with open(label_dst, "w") as f:
                for shape in data["shapes"]:
                    label = classes.index(shape["label"])
                    points = shape["points"]
                    img_width = data["imageWidth"]
                    img_height = data["imageHeight"]

                    x1 = min(points[0][0], points[1][0])
                    y1 = min(points[0][1], points[1][1])
                    x2 = max(points[0][0], points[1][0])
                    y2 = max(points[0][1], points[1][1])

                    width = x2 - x1
                    height = y2 - y1
                    xc = x1 + width / 2
                    yc = y1 + height / 2

                    xc /= img_width
                    width /= img_width
                    yc /= img_height
                    height /= img_height

                    f.write(f"{label} {xc} {yc} {width} {height}\n")

        except Exception as exp:
            CONSOLE.print(f"[red][ ERROR ] : {exp}[/red]")
            continue

        converted += 1

    CONSOLE.print(f"[cyan][ INFO ] : Converted {converted} out of {total} labels[/cyan]")
    CONSOLE.print(f"[cyan][ INFO ] : Copied {copied} out of {total} images[/cyan]")
    CONSOLE.print(f"[cyan][ INFO ] : Found {len(classes)} classes[/cyan]")


# Function to split images into train and test directories
def split_data(split: float = 0.8):
    input_loc = os.path.join(DATA_PATH, "all")
    train_image_loc = os.path.join(DATA_PATH, "train/images")
    train_label_loc = os.path.join(DATA_PATH, "train/labels")
    test_image_loc = os.path.join(DATA_PATH, "test/images")
    test_label_loc = os.path.join(DATA_PATH, "test/labels")

    try:
        if not os.path.exists(train_image_loc):
            os.makedirs(train_image_loc)

        if not os.path.exists(train_label_loc):
            os.makedirs(train_label_loc)
    except OSError as exp:
        CONSOLE.print(f"[red][ ERROR ] : {exp}[/red]")

    try:
        if not os.path.exists(test_image_loc):
            os.makedirs(test_image_loc)

        if not os.path.exists(test_label_loc):
            os.makedirs(test_label_loc)
    except OSError as exp:
        CONSOLE.print(f"[red][ ERROR ] : {exp}[/red]")

    images = [img for img in glob.glob(f"{input_loc}/images/*.jpg")]
    total = len(images)
    copied = 0
    train = 0
    test = 0

    for image in track(images, description="Splitting images..."):
        filename = os.path.splitext(os.path.basename(image))[0]

        img_src = os.path.join(input_loc, f"images/{filename}.jpg")
        label_src = os.path.join(input_loc, f"labels/{filename}.txt")

        if copied < total * split:
            img_dst = os.path.join(train_image_loc, f"{filename}.jpg")
            label_dst = os.path.join(train_label_loc, f"{filename}.txt")

            try:
                shutil.copy(img_src, img_dst)
                shutil.copy(label_src, label_dst)

                train += 1
                copied += 1
            except Exception as exp:
                CONSOLE.print(f"[red][ ERROR ] : {exp}[/red]")
        else:
            img_dst = os.path.join(test_image_loc, f"{filename}.jpg")
            label_dst = os.path.join(test_label_loc, f"{filename}.txt")

            try:
                shutil.copy(img_src, img_dst)
                shutil.copy(label_src, label_dst)

                test += 1
                copied += 1
            except Exception as exp:
                CONSOLE.print(f"[red][ ERROR ] : {exp}[/red]")

    CONSOLE.print(
        f"[cyan][ INFO ] : Copied {train} out of {total} images to train directory[/cyan]"
    )
    CONSOLE.print(f"[cyan][ INFO ] : Copied {test} out of {total} images to test directory[/cyan]")
    CONSOLE.print(f"[cyan][ INFO ] : Copied {copied} out of {total} images[/cyan]")


# Function to train YOLOv8 model
def train_model(weight: str = "yolov8n.pt", data: str = "data.yaml", epoch: int = 100):
    weights_loc = os.path.join(MODEL_WEIGHT_PATH, weight)
    data_loc = os.path.join(DATA_PATH, data)
    print(weights_loc)
    print(data_loc)

    # Check if weights file exists
    if not os.path.exists(weights_loc):
        CONSOLE.print("[red][ ERROR ] : Weights file not found[/red]")
        return

    # Check if data file exists
    if not os.path.exists(data_loc):
        CONSOLE.print("[red][ ERROR ] : Data file not found[/red]")
        return

    # Train YOLOv8 model
    CONSOLE.print("[cyan][ INFO ] : Training YOLOv8 model...[/cyan]")

    model = YOLO(weights_loc)
    model.train(data=data_loc, epochs=epoch)

    # Test YOLOv8 model
    CONSOLE.print("[cyan][ INFO ] : Testing YOLOv8 model...[/cyan]")
    metrics = model.val()

    CONSOLE.print(f"[cyan][ INFO ] : \n{metrics}[/cyan]")

    # Save trained model
    CONSOLE.print("[cyan][ INFO ] : Saving trained model...[/cyan]")
    success = model.export(format="onnx")
