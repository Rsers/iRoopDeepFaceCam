import os
import shutil
import uuid
import threading # Added
from flask import Flask, request, jsonify, send_from_directory, render_template # Added render_template

import modules.globals
import modules.core
from modules.processors.frame.core import get_frame_processors_modules
from modules.face_analyser import initialize_face_analyser
from modules.utilities import (has_image_extension, is_image, is_video, # Added is_video
                               extract_frames, get_temp_frame_paths, # Added video utils
                               create_video, restore_audio, create_temp,
                               move_temp, clean_temp, detect_fps)

app = Flask(__name__)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS.union({'mp4', 'avi', 'mov', 'mkv', 'webm'})

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_SOURCE_FOLDER = os.path.join(APP_ROOT, 'uploads/source_images')
UPLOAD_TARGET_FOLDER = os.path.join(APP_ROOT, 'uploads/target_media')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'outputs')

os.makedirs(UPLOAD_SOURCE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_TARGET_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

# --- process_video_task Function ---
def process_video_task(task_id, full_target_path_for_task, full_source_path_for_task, output_path_for_task, options_dict):
    # Target path for this specific task, to be used for clean_temp
    # This is necessary because modules.globals.target_path might be overwritten by another request
    # before this thread finishes.
    video_task_target_path = full_target_path_for_task

    print(f"Thread {threading.get_ident()}: Starting video processing for task {task_id}.")
    try:
        # Set globals for this thread's processing context
        # It's important that these are set here for thread-safety if the underlying modules use them directly
        # For modules that take paths as arguments, this might be less critical, but good practice.
        thread_globals = modules.globals # Create a reference for this thread if needed, though direct modification is usually fine for this app structure.

        thread_globals.source_path = full_source_path_for_task
        thread_globals.target_path = full_target_path_for_task
        thread_globals.output_path = output_path_for_task

        thread_globals.frame_processors = options_dict.get('frame_processors', ['face_swapper'])
        if not hasattr(thread_globals, 'fp_ui'): # Ensure fp_ui exists
            thread_globals.fp_ui = {}

        if 'face_enhancer' in thread_globals.frame_processors :
            thread_globals.fp_ui['face_enhancer'] = True
        else :
            thread_globals.fp_ui['face_enhancer'] = False

        thread_globals.keep_fps = bool(options_dict.get('keep_fps', True))
        thread_globals.keep_audio = bool(options_dict.get('keep_audio', True))
        thread_globals.keep_frames = bool(options_dict.get('keep_frames', False))
        thread_globals.many_faces = bool(options_dict.get('many_faces', False))

        exec_providers_req = options_dict.get('execution_provider', ['cpu'])
        if isinstance(exec_providers_req, str):
            exec_providers_req = [exec_providers_req]
        thread_globals.execution_providers = modules.core.decode_execution_providers(exec_providers_req)

        suggested_threads = modules.core.suggest_execution_threads()
        thread_globals.execution_threads = int(options_dict.get('execution_threads', suggested_threads))
        thread_globals.headless = True

        thread_globals.both_faces = bool(options_dict.get('both_faces', False))
        thread_globals.flip_faces = bool(options_dict.get('flip_faces', False))
        thread_globals.detect_face_right = bool(options_dict.get('detect_face_right', False))
        thread_globals.mouth_mask = bool(options_dict.get('mouth_mask', False))
        thread_globals.face_tracking = bool(options_dict.get('face_tracking', False))
        thread_globals.face_rot_range = int(options_dict.get('face_rot_range', 0))
        thread_globals.mask_size = int(options_dict.get('mask_size', 1))
        thread_globals.mask_down_size = float(options_dict.get('mask_down_size', 0.5))
        thread_globals.mask_feather_ratio = int(options_dict.get('mask_feather_ratio', 8))
        thread_globals.use_pseudo_face = bool(options_dict.get('use_pseudo_face', False))
        thread_globals.sticky_face_value = float(options_dict.get('sticky_face_value', 0.2))
        thread_globals.pseudo_face_threshold = float(options_dict.get('pseudo_face_threshold', 0.2))
        thread_globals.embedding_weight_size = float(options_dict.get('embedding_weight_size', 0.6))
        thread_globals.weight_distribution_size = float(options_dict.get('weight_distribution_size', 1.0))
        thread_globals.position_size = float(options_dict.get('position_size', 0.4))
        thread_globals.old_embedding_weight = float(options_dict.get('old_embedding_weight', 0.9))
        thread_globals.new_embedding_weight = float(options_dict.get('new_embedding_weight', 0.1))
        thread_globals.video_encoder = options_dict.get('video_encoder', 'libx264')
        thread_globals.video_quality = int(options_dict.get('video_quality', 18))
        thread_globals.show_target_face_box = bool(options_dict.get('show_target_face_box', False))
        thread_globals.show_mouth_mask_box = bool(options_dict.get('show_mouth_mask_box', False))

        initialize_face_analyser() # This should be thread-safe or initialize once globally
        modules.core.limit_resources() # This might need to be thread-aware if it sets process-wide limits

        active_processors = get_frame_processors_modules(thread_globals.frame_processors)
        for proc_module in active_processors:
            if not proc_module.pre_check():
                raise Exception(f"Pre-check failed for {proc_module.NAME} in video task {task_id}")
            if hasattr(proc_module, 'pre_start') and callable(getattr(proc_module, 'pre_start')) and not proc_module.pre_start():
                 raise Exception(f"Pre-start failed for {proc_module.NAME} in video task {task_id}")

        print(f"Thread {threading.get_ident()}: Video task {task_id}: Creating temp for {thread_globals.target_path}")
        create_temp(thread_globals.target_path)
        print(f"Thread {threading.get_ident()}: Video task {task_id}: Extracting frames")
        extract_frames(thread_globals.target_path, thread_globals.keep_fps) # Pass keep_fps here

        temp_frame_paths = get_temp_frame_paths(thread_globals.target_path)
        if not temp_frame_paths:
            raise Exception(f"No frames extracted for video task {task_id}")

        print(f"Thread {threading.get_ident()}: Video task {task_id}: Processing {len(temp_frame_paths)} frames")
        for frame_processor in active_processors:
            # Assuming process_video takes source_path and temp_frame_paths
            # And internally uses thread_globals for other settings if needed
            frame_processor.process_video(thread_globals.source_path, temp_frame_paths)

        modules.core.release_resources() # Call once after all processors for the video

        print(f"Thread {threading.get_ident()}: Video task {task_id}: Creating video output")
        # FPS for create_video is now handled inside extract_frames if keep_fps is true,
        # by storing it in a temp file. create_video will read it.
        # So, we don't need to pass fps directly to create_video here.
        create_video(thread_globals.target_path, thread_globals.keep_fps) # Pass keep_fps to create_video

        if thread_globals.keep_audio:
            print(f"Thread {threading.get_ident()}: Video task {task_id}: Restoring audio to {thread_globals.output_path}")
            restore_audio(thread_globals.target_path, thread_globals.output_path)
        else:
            print(f"Thread {threading.get_ident()}: Video task {task_id}: Moving temp video to {thread_globals.output_path}")
            move_temp(thread_globals.target_path, thread_globals.output_path)

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['output_filename'] = os.path.basename(thread_globals.output_path)
        print(f"Thread {threading.get_ident()}: Video processing for task {task_id} completed successfully.")

    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        import traceback
        traceback.print_exc()
        print(f"Thread {threading.get_ident()}: Video processing for task {task_id} failed: {e}")
    finally:
        print(f"Thread {threading.get_ident()}: Video task {task_id}: Cleaning temp for {video_task_target_path}")
        clean_temp(video_task_target_path)
        print(f"Thread {threading.get_ident()}: Video processing task {task_id} finished.")

# Original hello_world route removed as it's replaced by index route

@app.route('/upload/source_image', methods=['POST'])
def upload_source_image():
    if 'source_image' not in request.files:
        return jsonify({'error': 'No source_image part'}), 400
    file = request.files['source_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not (file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS)):
        return jsonify({'error': 'Invalid file type. Allowed image types: ' + ", ".join(sorted(list(ALLOWED_IMAGE_EXTENSIONS)))}), 400

    filename = file.filename # Consider using secure_filename from werkzeug.utils
    filepath = os.path.join(UPLOAD_SOURCE_FOLDER, filename)
    file.save(filepath)
    return jsonify({'message': 'Source image uploaded successfully', 'filepath': filename}), 200

@app.route('/upload/target_media', methods=['POST'])
def upload_target_media():
    if 'target_media' not in request.files:
        return jsonify({'error': 'No target_media part'}), 400
    file = request.files['target_media']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not (file and allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS)):
        return jsonify({'error': 'Invalid file type. Allowed media types: ' + ", ".join(sorted(list(ALLOWED_MEDIA_EXTENSIONS)))}), 400

    filename = file.filename # Consider using secure_filename from werkzeug.utils
    filepath = os.path.join(UPLOAD_TARGET_FOLDER, filename)
    file.save(filepath)
    return jsonify({'message': 'Target media uploaded successfully', 'filepath': filename}), 200

@app.route('/process', methods=['POST'])
def process_media():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    source_filename = data.get('source_path')
    target_filename = data.get('target_path')

    if not source_filename or not target_filename:
        return jsonify({'error': 'source_path (filename) and target_path (filename) are required'}), 400

    full_source_path = os.path.join(UPLOAD_SOURCE_FOLDER, source_filename)
    full_target_path = os.path.join(UPLOAD_TARGET_FOLDER, target_filename)

    if not os.path.exists(full_source_path):
        return jsonify({'error': f'Source file not found: {full_source_path}'}), 400
    if not os.path.exists(full_target_path):
        return jsonify({'error': f'Target file not found: {full_target_path}'}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending', 'source_filename': source_filename, 'target_filename': target_filename}

    options_dict = data.copy()

    base, ext = os.path.splitext(target_filename)
    output_filename_for_task = f"{base}_{task_id}_swapped{ext}"
    full_output_path_for_task = os.path.join(OUTPUT_FOLDER, output_filename_for_task)

    # Ensure fp_ui exists in modules.globals before any processing
    if not hasattr(modules.globals, 'fp_ui'):
        modules.globals.fp_ui = {}

    if has_image_extension(full_target_path):
        try:
            # IMAGE PROCESSING (Main Thread)
            # Set globals for this specific image processing task
            modules.globals.source_path = full_source_path
            modules.globals.target_path = full_target_path
            modules.globals.output_path = full_output_path_for_task

            modules.globals.frame_processors = options_dict.get('frame_processors', ['face_swapper'])
            if 'face_enhancer' in modules.globals.frame_processors :
                modules.globals.fp_ui['face_enhancer'] = True
            else :
                modules.globals.fp_ui['face_enhancer'] = False

            modules.globals.keep_fps = bool(options_dict.get('keep_fps', True)) # Not directly used for images
            modules.globals.keep_audio = bool(options_dict.get('keep_audio', True)) # Not directly used for images
            modules.globals.keep_frames = bool(options_dict.get('keep_frames', False))
            modules.globals.many_faces = bool(options_dict.get('many_faces', False))

            exec_providers_req = options_dict.get('execution_provider', ['cpu'])
            if isinstance(exec_providers_req, str):
                exec_providers_req = [exec_providers_req]
            modules.globals.execution_providers = modules.core.decode_execution_providers(exec_providers_req)

            suggested_threads = modules.core.suggest_execution_threads()
            modules.globals.execution_threads = int(options_dict.get('execution_threads', suggested_threads))
            modules.globals.headless = True

            modules.globals.both_faces = bool(options_dict.get('both_faces', False))
            modules.globals.flip_faces = bool(options_dict.get('flip_faces', False))
            modules.globals.detect_face_right = bool(options_dict.get('detect_face_right', False))
            modules.globals.mouth_mask = bool(options_dict.get('mouth_mask', False))
            modules.globals.face_tracking = bool(options_dict.get('face_tracking', False))
            modules.globals.face_rot_range = int(options_dict.get('face_rot_range', 0))
            modules.globals.mask_size = int(options_dict.get('mask_size', 1))
            modules.globals.mask_down_size = float(options_dict.get('mask_down_size', 0.5))
            modules.globals.mask_feather_ratio = int(options_dict.get('mask_feather_ratio', 8))
            modules.globals.use_pseudo_face = bool(options_dict.get('use_pseudo_face', False))
            modules.globals.sticky_face_value = float(options_dict.get('sticky_face_value', 0.2))
            modules.globals.pseudo_face_threshold = float(options_dict.get('pseudo_face_threshold', 0.2))
            modules.globals.embedding_weight_size = float(options_dict.get('embedding_weight_size', 0.6))
            modules.globals.weight_distribution_size = float(options_dict.get('weight_distribution_size', 1.0))
            modules.globals.position_size = float(options_dict.get('position_size', 0.4))
            modules.globals.old_embedding_weight = float(options_dict.get('old_embedding_weight', 0.9))
            modules.globals.new_embedding_weight = float(options_dict.get('new_embedding_weight', 0.1))
            modules.globals.show_target_face_box = bool(options_dict.get('show_target_face_box', False))
            modules.globals.show_mouth_mask_box = bool(options_dict.get('show_mouth_mask_box', False))
            # video_encoder and video_quality are not used for images, but set them from options if present
            modules.globals.video_encoder = options_dict.get('video_encoder', 'libx264')
            modules.globals.video_quality = int(options_dict.get('video_quality', 18))

            tasks[task_id]['status'] = 'processing_image'

            initialize_face_analyser()
            modules.core.limit_resources()

            active_processors = get_frame_processors_modules(modules.globals.frame_processors)
            for proc_module in active_processors:
                if not proc_module.pre_check():
                    raise Exception(f"Pre-check failed for {proc_module.NAME}")
                if hasattr(proc_module, 'pre_start') and callable(getattr(proc_module, 'pre_start')) and not proc_module.pre_start():
                    raise Exception(f"Pre-start failed for {proc_module.NAME}")

            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
            for frame_processor in active_processors:
                frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
                # modules.core.release_resources() # Typically called once after all processors for an image or video
            modules.core.release_resources() # Call once after all image processors completed

            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['output_filename'] = output_filename_for_task
            return jsonify({'message': 'Image processing completed', 'task_id': task_id, 'output_filename': output_filename_for_task}), 200

        except Exception as e:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = str(e)
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e), 'task_id': task_id}), 500

    elif is_video(full_target_path):
        tasks[task_id]['status'] = 'processing_video_queued'
        # Pass copies of path strings and options_dict to the thread
        thread = threading.Thread(target=process_video_task,
                                  args=(task_id,
                                        str(full_target_path),
                                        str(full_source_path),
                                        str(full_output_path_for_task),
                                        options_dict.copy()))
        thread.daemon = True
        thread.start()
        return jsonify({'message': 'Video processing started in background', 'task_id': task_id}), 202

    else:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = 'Target file is not a recognized image or video format.'
        return jsonify({'error': tasks[task_id]['error']}), 400

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task), 200

@app.route('/download/<task_id>', methods=['GET'])
def download_result(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    if task.get('status') != 'completed':
        # Allow download if video processing failed, to potentially get partial results or logs if implemented
        # For now, strictly completed tasks
        return jsonify({'error': 'Task not yet completed or failed'}), 400

    output_filename = task.get('output_filename')
    if not output_filename:
        return jsonify({'error': 'Output filename not found for task'}), 404

    return send_from_directory(OUTPUT_FOLDER, output_filename, as_attachment=True)

if __name__ == '__main__':
    # Global initializations
    if not hasattr(modules.globals, 'execution_providers') or not modules.globals.execution_providers:
        modules.globals.execution_providers = modules.core.decode_execution_providers(['cpu'])
    if not hasattr(modules.globals, 'fp_ui'): # Ensure fp_ui exists globally
        modules.globals.fp_ui = {}

    app.run(debug=True, host='0.0.0.0', port=5000)
