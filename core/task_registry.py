from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from PySide6.QtCore import QT_TRANSLATE_NOOP
except ImportError:
    def QT_TRANSLATE_NOOP(_context: str, source: str) -> str:
        return source


@dataclass(frozen=True, slots=True)
class TaskSettingSpec:
    setting_id: str
    json_key: str
    default: Any
    value_type: type[Any]
    options: tuple[Any, ...] | None = None
    label: str | None = None
    description: str | None = None
    visible: bool = True
    coerce: Callable[[Any], Any] | None = field(default=None, repr=False, compare=False)


def _task_setting(
    setting_id: str,
    json_key: str,
    default: Any,
    value_type: type[Any],
    *,
    options: tuple[Any, ...] | None = None,
    label: str | None = None,
    description: str | None = None,
    visible: bool = True,
    coerce: Callable[[Any], Any] | None = None,
) -> TaskSettingSpec:
    return TaskSettingSpec(
        setting_id=setting_id,
        json_key=json_key,
        default=default,
        value_type=value_type,
        options=options,
        label=label,
        description=description,
        visible=visible,
        coerce=coerce,
    )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    key: str
    title: str
    description: str
    module_path: str
    class_name: str
    default_params: dict[str, Any] = field(default_factory=dict)
    settings: tuple[TaskSettingSpec, ...] = ()


_TASK_SPECS = (
    TaskSpec(
        key='files-renamer',
        title=QT_TRANSLATE_NOOP('Tasks', 'Batch Rename'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Batch rename files using prefix / all / body / between rules.'),
        module_path='modules.files_renamer',
        class_name='FilesRenamer',
        settings=(
            _task_setting('rename_log_folder_name', 'log_folder_name', 'files_renamer_log', str),
            _task_setting(
                'rename_mode',
                'mode',
                'prefix',
                str,
                options=('prefix', 'all', 'body', 'between'),
            ),
            _task_setting('rename_pattern', 'pattern', '-', str),
            _task_setting('rename_start_pattern', 'start_pattern', '', str),
            _task_setting('rename_end_pattern', 'end_pattern', '', str),
            _task_setting('rename_replace_with', 'replace_with', '', str),
            _task_setting('rename_include_extension', 'include_extension', False, bool, options=(True, False)),
            _task_setting('rename_case_sensitive', 'case_sensitive', False, bool, options=(True, False)),
            _task_setting('rename_recursive', 'recursive', False, bool, options=(True, False)),
            _task_setting('rename_max_threads', 'max_threads', 4, int, options=(1, 2, 4, 8)),
        ),
    ),
    TaskSpec(
        key='bilibili-export',
        title=QT_TRANSLATE_NOOP('Tasks', 'Bilibili Video Export'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Batch repair and merge cached Bilibili videos into playable MP4 files.'),
        module_path='modules.bili_videos',
        class_name='BiliVideos',
        settings=(
            _task_setting('bili_log_folder_name', 'log_folder_name', 'bili_video_handle_log', str),
            _task_setting('bili_out_dir_prefix', 'out_dir_prefix', 'fixed-', str),
            _task_setting('bili_add_group_title', 'AddGroupTitle', True, bool, options=(True, False)),
            _task_setting(
                'bili_group_title_max_length',
                'GroupTitleMaxLength',
                10,
                int,
                options=(5, 10, 15, 20),
            ),
        ),
    ),
    TaskSpec(
        key='subtitle-generation',
        title=QT_TRANSLATE_NOOP('Tasks', 'Subtitle Generation'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Batch extract audio from media files and generate SRT subtitles.'),
        module_path='modules.gen_subtitles',
        class_name='GenSubtitles',
        settings=(
            _task_setting('gen_subtitle_log_folder_name', 'log_folder_name', 'gen_subtitles_log', str),
            _task_setting('gen_subtitle_model_path', 'model_path', '', str),
            _task_setting('gen_subtitle_parallel', 'parallel', False, bool, options=(True, False)),
        ),
    ),
    TaskSpec(
        key='mac-cleaner',
        title=QT_TRANSLATE_NOOP('Tasks', 'Mac Cleaner'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Batch clean system junk files under the selected directories.'),
        module_path='modules.mac_poop_scooper',
        class_name='MacPoopScooper',
    ),
    TaskSpec(
        key='merge-colors',
        title=QT_TRANSLATE_NOOP('Tasks', 'Merge Color Channels'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Pair R/G/B images by filename prefix and merge them into composite color images.'),
        module_path='modules.merge_colors',
        class_name='MergeColors',
        default_params={'colors': ['R', 'G']},
        settings=(
            _task_setting('mergecolor_log_folder_name', 'log_folder_name', 'merge_color_log', str),
            _task_setting('mergecolor_out_dir_prefix', 'out_dir_prefix', 'merge-', str),
        ),
    ),
    TaskSpec(
        key='split-colors',
        title=QT_TRANSLATE_NOOP('Tasks', 'Split Color Channels'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Split input images into separate R/G/B channels.'),
        module_path='modules.split_colors',
        class_name='SplitColors',
        default_params={'colors': ['R', 'G']},
    ),
    TaskSpec(
        key='twist-images',
        title=QT_TRANSLATE_NOOP('Tasks', 'Image Perspective Transform'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Apply a perspective transform to images using preset quadrilateral parameters.'),
        module_path='modules.twist_shape',
        class_name='TwistImgs',
        default_params={'twisted_corner': [[0, 0], [430, 82], [432, 268], [0, 276]]},
    ),
    TaskSpec(
        key='ecg-handler',
        title=QT_TRANSLATE_NOOP('Tasks', 'ECG Signal Processing'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Analyze ECG CSV data and generate raw, filtered, and advanced analysis charts.'),
        module_path='modules.ECG_handler',
        class_name='ECGHandler',
        settings=(
            _task_setting('ecg_log_folder_name', 'log_folder_name', 'ecg_handler_log', str),
            _task_setting('ecg_sampling_rate', 'sampling_rate', 1000, int, options=(100, 200, 500, 1000, 2000)),
            _task_setting('ecg_parallel', 'parallel', False, bool, options=(True, False)),
            _task_setting('ecg_out_dir_prefix', 'out_dir_prefix', 'ecg-', str),
            _task_setting('ecg_filter_low_cut', 'filter_low_cut', 0.5, float, options=(0.1, 0.5, 0.67, 0.8, 1.0, 2.0)),
            _task_setting('ecg_filter_high_cut', 'filter_high_cut', 30.0, float, options=(5.0, 15.0, 30.0, 50.0, 100.0)),
            _task_setting('ecg_filter_order', 'filter_order', 2, int, options=(1, 2, 4)),
            _task_setting('ecg_drop_raw_zero', 'drop_raw_zero', False, bool, options=(True, False)),
            _task_setting('ecg_trim_raw_data', 'trim_raw_data', False, bool, options=(True, False)),
            _task_setting('ecg_trim_filtered_data', 'trim_filtered_data', True, bool, options=(True, False)),
            _task_setting('ecg_trim_percentage', 'trim_percentage', 6.0, float, options=(5.0, 6.0, 8.0, 10.0, 15.0, 20.0)),
            _task_setting('ecg_time_range_short', 'time_range_short', 3.0, float, options=(1.0, 2.0, 3.0, 4.0)),
            _task_setting('ecg_time_range_medium', 'time_range_medium', 10.0, float, options=(10.0, 15.0, 20.0, 30.0)),
        ),
    ),
    TaskSpec(
        key='dicom-processing',
        title=QT_TRANSLATE_NOOP('Tasks', 'DICOM Processing'),
        description=QT_TRANSLATE_NOOP('Tasks', 'Batch read DICOM series, export images, and generate videos when needed.'),
        module_path='modules.dicom_to_imgs',
        class_name='DicomToImage',
        settings=(
            _task_setting('dicom_log_folder_name', 'log_folder_name', 'dicom_handle_log', str),
            _task_setting('dicom_fps', 'fps', 10, int, options=(10, 20, 30)),
            _task_setting('dicom_frame_dpi', 'frame_dpi', 200, int, options=(100, 200, 400, 800)),
            _task_setting('dicom_out_dir_prefix', 'out_dir_prefix', 'Img-', str),
        ),
    ),
)

_TASK_SPEC_BY_KEY = {spec.key: spec for spec in _TASK_SPECS}


def get_task_specs() -> list[TaskSpec]:
    return list(_TASK_SPECS)


def get_task_spec(task_key: str) -> TaskSpec:
    return _TASK_SPEC_BY_KEY[task_key]
