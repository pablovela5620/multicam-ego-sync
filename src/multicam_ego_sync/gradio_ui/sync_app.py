import datetime
import subprocess
import sys
import uuid
import zipfile
from contextlib import nullcontext
from pathlib import Path
from timeit import default_timer as timer

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from gradio_rerun.events import TimeUpdate
from rich.console import Console
from serde import serde
from serde.json import to_json
from tqdm import tqdm

CONSOLE = Console(width=120)


@serde
class Info:
    capture_date: datetime.date
    ego_start_second: int | float
    multicam_start_second: int | float
    # realsense_start_second: Optional[float]


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def run_command(cmd: str, verbose=False) -> str | None:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule(
            "[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ",
            style="red",
        )
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


def convert_to_av1(input_video_path: Path, output_video_path: Path, verbose: bool = False) -> None:
    # check if the new video path already exists
    if output_video_path.exists():
        print(f"File {output_video_path} already exists")
        return

    convert_cmd = [
        f"ffmpeg ",  # noqa F541
        f"-i '{str(input_video_path)}' ",
        f"-c:v libsvtav1 ",  # noqa F541
        f"-preset {10} ",
        f"-crf {35} ",
        f"-c:a copy ",  # noqa F541
        f"{str(output_video_path)}",
    ]
    convert_cmd = " ".join(convert_cmd)
    with status(
        msg="[bold yellow]Running av1 conversion... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(convert_cmd, verbose=verbose)


def cut_video(
    input_path: str,
    output_path: str,
    start_time: float,
    duration: float,
    verbose: bool = False,
) -> None:
    """Cut video using ffmpeg with fast AV1 encoding."""
    start_trim_time = timer()
    cut_cmd = [
        f"ffmpeg ",  # noqa F541
        f"-y ",  # noqa F541
        f"-accurate_seek ",  # noqa F541
        f"-ss {start_time:.3f} ",
        f"-i '{input_path}' ",
        f"-t {duration:.3f} ",
        f"-fps_mode cfr ",  # noqa F541
        f"-c:v libsvtav1 ",  # noqa F541
        f"-preset {13} ",
        f"-crf {45} ",
        f"-c:a copy ",  # noqa F541
        f"{output_path}",
    ]
    cut_cmd = " ".join(cut_cmd)
    with status(
        msg="[bold yellow]Trimming Videos... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(cut_cmd, verbose=verbose)

    end_trim_time = timer()
    print(f"Time taken to cut video: {end_trim_time - start_trim_time}")


# Whenever we need a recording, we construct a new recording stream.
# As long as the app and recording IDs remain the same, the data
# will be merged by the Viewer.
def get_recording(recording_id: str) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


def unzip_multiview_capture(zip_path: str) -> Path:
    """
    Unzips the Multiview Capture folder into its parent directory, avoiding nested folders.
    Returns the path to the actual extracted content directory (e.g., parent/zip_stem).
    """
    zip_path: Path = Path(zip_path)
    if not zip_path.exists():
        raise gr.Error(f"File {zip_path} does not exist")
    if not zipfile.is_zipfile(zip_path):
        raise gr.Error(f"File {zip_path} is not a valid zip file.")

    # Define the target directory for extraction as the parent of the zip file
    extract_target_dir = zip_path.parent
    extract_target_dir.mkdir(parents=True, exist_ok=True)  # Ensure parent exists

    # Define the expected path of the main content directory after extraction
    # Assumes the zip file contains a top-level folder named like the zip file itself (without .zip)
    expected_content_dir = extract_target_dir / zip_path.stem

    # Check if the content directory already exists and has content
    if expected_content_dir.is_dir() and any(expected_content_dir.iterdir()):
        print(f"Content directory {expected_content_dir} already exists and is not empty. Skipping extraction.")
        return expected_content_dir

    # Extract the zip file into the parent directory
    print(f"Extracting {zip_path} to {extract_target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_target_dir)
    except zipfile.BadZipFile as e:
        raise gr.Error(f"File {zip_path} is not a valid zip file.") from e
    except Exception as e:
        raise gr.Error(f"An error occurred during extraction: {e}") from e

    # Verify extraction produced the expected directory
    if not expected_content_dir.is_dir():
        # If the zip didn't contain a root folder matching the stem, raise an error or adjust logic
        # For now, assume it should exist.
        raise gr.Error(f"Extraction failed or did not produce the expected directory: {expected_content_dir}")

    print(f"Extraction successful. Content directory: {expected_content_dir}")
    # Return the path to the actual extracted content directory
    return expected_content_dir


def log_video(rec: rr.RecordingStream, video_asset: rr.AssetVideo, name: str) -> None:
    rec.log(f"{name}", video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rec.send_columns(
        f"{name}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeColumn("video_time", duration=1e-9 * frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
    )


def track_current_time(evt: TimeUpdate):
    return evt.payload.time


def log_ego(recording_id: str, capture_dir: str):
    # Here we get a recording using the provided recording id.
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    capture_dir: Path = Path(capture_dir)
    assert capture_dir.exists(), f"{capture_dir} does not exist"

    raw_data_dir = capture_dir / "raw-capture"
    assert raw_data_dir.exists(), f"{raw_data_dir} is mandatory"

    ego_video_path = raw_data_dir / "ego.mp4"
    assert ego_video_path.exists(), f"{ego_video_path} is mandatory"

    # # keep the original video
    # preprocessed_dir = capture_dir / "preprocessed"
    # preprocessed_dir.mkdir(exist_ok=True, parents=True)

    # ego_output_path = preprocessed_dir / "ego-av1.mp4"

    # convert_to_av1(
    #     input_video_path=ego_video_path,
    #     output_video_path=ego_output_path,
    #     verbose=False,
    # )

    video_asset = rr.AssetVideo(path=ego_video_path)
    log_video(rec, video_asset=video_asset, name="ego")

    if capture_dir is None:
        raise gr.Error("Must provide an Multiview Capture folder")

    blueprint = rrb.Blueprint(
        collapse_panels=True,
    )

    rec.send_blueprint(blueprint)
    yield stream.read()


def log_exo(recording_id: str, capture_dir: str):
    # Here we get a recording using the provided recording id.
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    capture_dir: Path = Path(capture_dir)
    assert capture_dir.exists(), f"{capture_dir} does not exist"

    raw_data_dir = capture_dir / "raw-capture"
    assert raw_data_dir.exists(), f"{raw_data_dir} is mandatory"

    blueprint = rrb.Blueprint(
        collapse_panels=True,
    )

    multicam_dir = raw_data_dir / "exo-multicam"
    assert multicam_dir.exists(), f"{multicam_dir} is mandatory"

    multicam_path_list = list(multicam_dir.glob("*.mp4"))
    multicam_total_frames_list = []
    for idx, multicam_path in enumerate(tqdm(multicam_path_list)):
        video_asset = rr.AssetVideo(path=multicam_path)
        multicam_timestamps_ns = video_asset.read_frame_timestamps_nanos()
        multicam_total_frames_list.append(len(multicam_timestamps_ns))
        log_video(rec, video_asset=video_asset, name=f"multicam-{idx}")

    assert len(set(multicam_total_frames_list)) == 1, "All multicam videos should have the same number of frames"

    rec.send_blueprint(blueprint)
    yield stream.read()


def synchronize_capture(capture_dir: str, ego_start_time: int | float, exo_start_time: int | float):
    """
    Synchronizes the capture by adjusting the start times of the ego and exo videos.
    """
    rec: rr.RecordingStream = get_recording(uuid.uuid4())
    stream: rr.BinaryStream = rec.binary_stream()

    info: Info = Info(
        capture_date=datetime.datetime.now().replace(microsecond=0),
        ego_start_second=ego_start_time,
        multicam_start_second=exo_start_time,
    )
    info_json_str: str = to_json(info)

    capture_dir: Path = Path(capture_dir)
    raw_data_dir = capture_dir / "raw-capture"
    assert raw_data_dir.exists(), f"{raw_data_dir} is mandatory"

    ego_video_path = raw_data_dir / "ego.mp4"
    assert ego_video_path.exists(), f"{ego_video_path} is mandatory"

    # keep the original video
    preprocessed_dir = capture_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True, parents=True)

    ego_output_path = preprocessed_dir / "ego-av1.mp4"

    convert_to_av1(
        input_video_path=ego_video_path,
        output_video_path=ego_output_path,
        verbose=False,
    )

    ego_video_asset = rr.AssetVideo(path=ego_output_path)
    ego_timestamps_ns = ego_video_asset.read_frame_timestamps_nanos()
    # log_video(rec=rec, video_asset=ego_video_asset, name="ego")
    yield stream.read(), info_json_str

    multicam_dir = raw_data_dir / "exo-multicam"
    assert multicam_dir.exists(), f"{multicam_dir} is mandatory"

    multicam_path_list = list(multicam_dir.glob("*.mp4"))
    multicam_output_paths: list[Path] = [
        preprocessed_dir / f"multicam-av1-{idx}.mp4" for idx in range(len(multicam_path_list))
    ]

    multicam_total_frames_list = []
    for idx, multicam_path in enumerate(tqdm(multicam_path_list)):
        multicam_output_path: Path = preprocessed_dir / f"multicam-av1-{idx}.mp4"
        convert_to_av1(
            input_video_path=multicam_path,
            output_video_path=multicam_output_path,
            verbose=False,
        )
        video_asset = rr.AssetVideo(path=multicam_output_path)
        multicam_timestamps_ns = video_asset.read_frame_timestamps_nanos()
        multicam_total_frames_list.append(len(multicam_timestamps_ns))
        # log_video(rec, video_asset=video_asset, name=f"multicam-{idx}")
        yield stream.read(), info_json_str

    # cut video such that the start frame is chosen and then end frame is the shortest of the two
    ego_end_second = ego_timestamps_ns[-1] / 1e9 - info.ego_start_second
    multicam_end_second = multicam_timestamps_ns[-1] / 1e9 - info.multicam_start_second
    smallest_end_second = min(ego_end_second, multicam_end_second) - 1

    # make directory for synchronized videos
    synchronized_dir = capture_dir / "synchronized"
    synchronized_dir.mkdir(exist_ok=True, parents=True)

    synchronized_videos = []

    # cut and log the ego video
    final_ego_path = synchronized_dir / ego_output_path.name.replace("av1", "cut")
    print(f"final_ego_output_path: {final_ego_path}")
    # delete the file if it already exists
    if final_ego_path.exists():
        final_ego_path.unlink()
    cut_video(
        input_path=str(ego_output_path),
        output_path=str(final_ego_path),
        start_time=info.ego_start_second,
        duration=smallest_end_second,
        verbose=False,
    )
    ego_video_asset = rr.AssetVideo(path=final_ego_path)
    log_video(rec, video_asset=ego_video_asset, name="ego")
    yield stream.read(), info_json_str

    # cut each video to start frames and shortest num frames
    for idx, multicam_output_path in enumerate(multicam_output_paths):
        final_sync_path = synchronized_dir / multicam_output_path.name.replace("av1", "cut")
        print(f"final_multicam_output_path: {final_sync_path}")
        # delete the file if it already exists
        if final_sync_path.exists():
            final_sync_path.unlink()

        cut_video(
            input_path=str(multicam_output_path),
            output_path=str(final_sync_path),
            start_time=info.multicam_start_second,
            duration=smallest_end_second,
            verbose=False,
        )
        synchronized_videos.append(final_sync_path)
        video_asset = rr.AssetVideo(path=final_sync_path)
        log_video(rec, video_asset=video_asset, name=f"multicam-{idx}")
        yield stream.read(), info_json_str

    print(f"Synchronized dir: {synchronized_dir}")
    # save json str to file
    json_path = synchronized_dir / "info.json"
    with open(json_path, "w") as f:
        f.write(info_json_str)
    print(f"Saved info.json to {json_path}")

    assert len(set(multicam_total_frames_list)) == 1, "All multicam videos should have the same number of frames"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="ego"),
            rrb.Grid(
                contents=[rrb.Spatial2DView(origin=f"multicam-{idx}") for idx in range(len(multicam_path_list))],
            ),
        ),
        collapse_panels=True,
    )
    rec.send_blueprint(blueprint)

    yield stream.read(), to_json(info)


def zip_directory(capture_root_dir: Path) -> str:
    # print the current working directory
    print(f"Current working directory: {Path.cwd()}")
    synchronized_dir = capture_root_dir / "synchronized"
    if not synchronized_dir.is_dir():
        # It's good practice to raise an error if the directory doesn't exist.
        # Gradio can display this error to the user.
        raise gr.Error(f"Synchronized directory not found: {synchronized_dir}")

    # Name the output zip file to indicate it contains synchronized data.
    # It will be created in the parent directory of the capture_root_dir.
    output_zip_path: Path = capture_root_dir.parent / f"{capture_root_dir.name}-synchronized.zip"

    zip_cmd_str: str = (
        f"cd '{str(synchronized_dir.parent)}' && zip -r '{str(output_zip_path)}' '{synchronized_dir.name}'"
    )

    with status(
        msg="[bold yellow]Zipping synchronized capture folder... (This may take a while)",
        spinner="circle",
        verbose=False,
    ):
        run_command(zip_cmd_str, verbose=False)

    return str(output_zip_path)


def update_viewer_visibility() -> tuple[Rerun, Rerun, Rerun, gr.Tabs]:
    exo_viewer_update = Rerun(visible=False)
    ego_viewer_update = Rerun(visible=False)
    synchronized_viewer_update = Rerun(visible=True)
    tabs_update = gr.Tabs(selected="Output")  # Select the "Output" tab by its label
    return ego_viewer_update, exo_viewer_update, synchronized_viewer_update, tabs_update


with gr.Blocks() as sync_block:
    mv_dir: Path | gr.State = gr.State(Path())
    with gr.Tabs() as app_tabs:
        with gr.Tab("Input"):
            with gr.Row():
                mv_zip = gr.File(label="Upload Multiview Capture Folder", file_count="single", file_types=[".zip"])
                with gr.Column():
                    sync_btn = gr.Button("Synchronize Capture")
                    with gr.Row():
                        ego_time = gr.Number(label="Ego Time (Seconds)", interactive=False, precision=3)
                        exo_time = gr.Number(label="Exo Time (Seconds)", interactive=False, precision=3)
        with gr.Tab("Output"):
            output_zip = gr.File(label="Final Output", file_count="single")
            output_json = gr.JSON(label="Output JSON")
    with gr.Row():
        ego_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
            visible=True,
        )
        exo_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
            visible=True,
        )
        synchronized_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
            visible=False,
        )

    # We make a new recording id, and store it in a Gradio's session state.
    ego_recording_id = gr.State(uuid.uuid4())
    exo_multicam_recording_id = gr.State(uuid.uuid4())

    # get start time from the rerun viewers using callbacks
    ego_viewer.time_update(
        fn=track_current_time,
        outputs=[ego_time],
    ).then(
        # convert from nanos to seconds
        fn=lambda time: time / 1e9,
        inputs=[ego_time],
        outputs=[ego_time],
    )

    exo_viewer.time_update(
        fn=track_current_time,
        outputs=[exo_time],
    ).then(
        # convert from nanos to seconds
        fn=lambda time: time / 1e9,
        inputs=[exo_time],
        outputs=[exo_time],
    )

    # on zipfile upload, extract the files and log the ego and exo videos
    mv_zip.upload(
        fn=unzip_multiview_capture,
        inputs=[mv_zip],
        outputs=[mv_dir],
    ).then(
        fn=log_exo,
        inputs=[exo_multicam_recording_id, mv_dir],
        outputs=[exo_viewer],
    ).then(
        fn=log_ego,
        inputs=[ego_recording_id, mv_dir],
        outputs=[ego_viewer],
    )

    # on synchronize button click, synchronize the capture
    sync_btn.click(
        fn=update_viewer_visibility,
        inputs=[],
        outputs=[ego_viewer, exo_viewer, synchronized_viewer, app_tabs],
    ).then(
        fn=synchronize_capture,
        inputs=[mv_dir, ego_time, exo_time],
        outputs=[synchronized_viewer, output_json],
    ).then(
        # zip mv directory and save it
        fn=zip_directory,
        inputs=[mv_dir],
        outputs=[output_zip],
    )
