from typing import Any, Dict, Optional

WORDING : Dict[str, Any] =\
{
	'conda_not_activated': 'Conda 环境未激活',
	'python_not_supported': 'Python版本不支持，请升级到 {version} 或更高版本',
	'curl_not_installed': 'CURL 未安装',
	'ffmpeg_not_installed': 'FFMpeg 未安装',
	'creating_temp': '正在创建临时资源',
	'extracting_frames': '正在提取帧，分辨率为 {resolution}，每秒 {fps} 帧',
	'extracting_frames_succeed': '帧提取成功',
	'extracting_frames_failed': '帧提取失败',
	'analysing': '分析中',
	'processing': '处理中',
	'downloading': '下载中',
	'temp_frames_not_found': '未找到临时帧',
	'copying_image': '正在复制图片，分辨率为 {resolution}',
	'copying_image_succeed': '图片复制成功',
	'copying_image_failed': '图片复制失败',
	'finalizing_image': '正在完成图片，分辨率为 {resolution}',
	'finalizing_image_succeed': '图片完成成功',
	'finalizing_image_skipped': '跳过图片完成',
	'merging_video': '正在合并视频，分辨率为 {resolution}，每秒 {fps} 帧',
	'merging_video_succeed': '视频合并成功',
	'merging_video_failed': '视频合并失败',
	'skipping_audio': '跳过音频',
	'replacing_audio_succeed': '音频替换成功',
	'replacing_audio_skipped': '跳过音频替换',
	'restoring_audio_succeed': '音频恢复成功',
	'restoring_audio_skipped': '跳过音频恢复',
	'clearing_temp': '正在清理临时资源',
	'processing_stopped': '处理已停止',
	'processing_image_succeed': '图片处理成功，用时 {seconds} 秒',
	'processing_image_failed': '图片处理失败',
	'processing_video_succeed': '视频处理成功，用时 {seconds} 秒',
	'processing_video_failed': '视频处理失败',
	'choose_image_source': '选择源图片',
	'choose_audio_source': '选择源音频',
	'choose_video_target': '选择目标视频',
	'choose_image_or_video_target': '选择目标图片或视频',
	'specify_image_or_video_output': '在目录中指定输出图片或视频',
	'match_target_and_output_extension': '目标和输出扩展名需要匹配',
	'no_source_face_detected': '未检测到源人脸',
	'processor_not_loaded': '处理器 {processor} 无法加载',
	'processor_not_implemented': '处理器 {processor} 未正确实现',
	'ui_layout_not_loaded': 'UI布局 {ui_layout} 无法加载',
	'ui_layout_not_implemented': 'UI布局 {ui_layout} 未正确实现',
	'stream_not_loaded': '流 {stream_mode} 无法加载',
	'job_created': '任务 {job_id} 已创建',
	'job_not_created': '任务 {job_id} 未创建',
	'job_submitted': '任务 {job_id} 已提交',
	'job_not_submitted': '任务 {job_id} 未提交',
	'job_all_submitted': '所有任务已提交',
	'job_all_not_submitted': '所有任务未提交',
	'job_deleted': '任务 {job_id} 已删除',
	'job_not_deleted': '任务 {job_id} 未删除',
	'job_all_deleted': '所有任务已删除',
	'job_all_not_deleted': '所有任务未删除',
	'job_step_added': '步骤已添加到任务 {job_id}',
	'job_step_not_added': '步骤未添加到任务 {job_id}',
	'job_remix_step_added': '已从任务 {job_id} 重混步骤 {step_index}',
	'job_remix_step_not_added': '未从任务 {job_id} 重混步骤 {step_index}',
	'job_step_inserted': '步骤 {step_index} 已插入到任务 {job_id}',
	'job_step_not_inserted': '步骤 {step_index} 未插入到任务 {job_id}',
	'job_step_removed': '已从任务 {job_id} 移除步骤 {step_index}',
	'job_step_not_removed': '未从任务 {job_id} 移除步骤 {step_index}',
	'running_job': '正在运行队列中的任务 {job_id}',
	'running_jobs': '正在运行所有队列中的任务',
	'retrying_job': '正在重试失败的任务 {job_id}',
	'retrying_jobs': '正在重试所有失败的任务',
	'processing_job_succeed': '任务 {job_id} 处理成功',
	'processing_jobs_succeed': '所有任务处理成功',
	'processing_job_failed': '任务 {job_id} 处理失败',
	'processing_jobs_failed': '所有任务处理失败',
	'processing_step': '正在处理步骤 {step_current}/{step_total}',
	'validating_hash_succeed': '{hash_file_name} 的哈希验证成功',
	'validating_hash_failed': '{hash_file_name} 的哈希验证失败',
	'validating_source_succeed': '{source_file_name} 的源验证成功',
	'validating_source_failed': '{source_file_name} 的源验证失败',
	'deleting_corrupt_source': '正在删除损坏的源文件 {source_file_name}',
	'time_ago_now': '刚刚',
	'time_ago_minutes': '{minutes} 分钟前',
	'time_ago_hours': '{hours} 小时 {minutes} 分钟前',
	'time_ago_days': '{days} 天 {hours} 小时 {minutes} 分钟前',
	'point': '。',
	'comma': '，',
	'colon': '：',
	'question_mark': '？',
	'exclamation_mark': '！',
	'about':
	{
		'become_a_member': '成为会员',
		'join_our_community': '加入我们的社区',
		'read_the_documentation': '阅读文档'
	},
	'help':
	{
		# installer
		'install_dependency': '选择要安装的 {dependency} 版本',
		'skip_conda': '跳过 conda 环境检查',
		# paths
		'config_path': '选择配置文件以覆盖默认设置',
		'jobs_path': '指定存储任务的目录',
		'source_paths': '选择单个或多个源图片或音频',
		'target_path': '选择单个目标图片或视频',
		'output_path': '在目录中指定输出图片或视频',
		# face detector
		'face_detector_model': '选择负责检测人脸的模型',
		'face_detector_size': '指定提供给人脸检测器的帧大小',
		'face_detector_angles': '指定在检测人脸前旋转帧的角度',
		'face_detector_score': '基于置信度分数过滤检测到的人脸',
		# face landmarker
		'face_landmarker_model': '选择负责检测人脸特征点的模型',
		'face_landmarker_score': '基于置信度分数过滤检测到的人脸特征点',
		# face selector
		'face_selector_mode': '使用基于参考的跟踪或简单匹配',
		'face_selector_order': '指定检测到的人脸的顺序',
		'face_selector_age_start': '基于起始年龄过滤检测到的人脸',
		'face_selector_age_end': '基于结束年龄过滤检测到的人脸',
		'face_selector_gender': '基于性别过滤检测到的人脸',
		'face_selector_race': '基于种族过滤检测到的人脸',
		'reference_face_position': '指定用于创建参考人脸的位置',
		'reference_face_distance': '指定参考人脸和目标人脸之间的相似度',
		'reference_frame_number': '指定用于创建参考人脸的帧',
		# face masker
		'face_mask_types': '混合搭配不同的人脸遮罩类型 (选项: {choices})',
		'face_mask_blur': '指定应用于方框遮罩的模糊程度',
		'face_mask_padding': '为方框遮罩应用上、右、下、左填充',
		'face_mask_regions': '选择用于区域遮罩的面部特征 (选项: {choices})',
		# frame extraction
		'trim_frame_start': '指定目标视频的起始帧',
		'trim_frame_end': '指定目标视频的结束帧',
		'temp_frame_format': '指定临时资源格式',
		'keep_temp': '处理后保留临时资源',
		# output creation
		'output_image_quality': '指定图片质量(影响压缩率)',
		'output_image_resolution': '基于目标图片指定输出图片分辨率',
		'output_audio_encoder': '指定用于音频输出的编码器',
		'output_video_encoder': '指定用于视频输出的编码器',
		'output_video_preset': '平衡视频处理速度和文件大小',
		'output_video_quality': '指定视频质量(影响压缩率)',
		'output_video_resolution': '基于目标视频指定输出视频分辨率',
		'output_video_fps': '基于目标视频指定输出视频帧率',
		'skip_audio': '从目标视频中省略音频',
		# processors
		'processors': '加载单个或多个处理器 (选项: {choices}, ...)',
		'age_modifier_model': '选择负责调整年龄的模型',
		'age_modifier_direction': '指定年龄修改的方向',
		'expression_restorer_model': '选择负责恢复表情的模型',
		'expression_restorer_factor': '从目标人脸恢复表情的因子',
		'face_debugger_items': '加载单个或多个处理器 (选项: {choices})',
		'face_editor_model': '选择负责编辑人脸的模型',
		'face_editor_eyebrow_direction': '指定眉毛方向',
		'face_editor_eye_gaze_horizontal': '指定水平眼睛视线',
		'face_editor_eye_gaze_vertical': '指定垂直眼睛视线',
		'face_editor_eye_open_ratio': '指定眼睛开启比例',
		'face_editor_lip_open_ratio': '指定嘴唇开启比例',
		'face_editor_mouth_grim': '指定嘴巴严肃程度',
		'face_editor_mouth_pout': '指定嘴巴撅起程度',
		'face_editor_mouth_purse': '指定嘴巴抿起程度',
		'face_editor_mouth_smile': '指定嘴巴微笑程度',
		'face_editor_mouth_position_horizontal': '指定嘴巴水平位置',
		'face_editor_mouth_position_vertical': '指定嘴巴垂直位置',
		'face_editor_head_pitch': '指定头部俯仰角度',
		'face_editor_head_yaw': '指定头部偏航角度',
		'face_editor_head_roll': '指定头部翻滚角度',
		'face_enhancer_model': '选择负责增强人脸的模型',
		'face_enhancer_blend': '将增强后的人脸混合到原始人脸',
		'face_swapper_model': '选择负责换脸的模型',
		'face_swapper_pixel_boost': '选择换脸的像素提升分辨率',
		'frame_colorizer_model': '选择负责为帧上色的模型',
		'frame_colorizer_size': '指定提供给帧上色器的帧大小',
		'frame_colorizer_blend': '将上色后的帧混合到原始帧',
		'frame_enhancer_model': '选择负责增强帧的模型',
		'frame_enhancer_blend': '将增强后的帧混合到原始帧',
		'lip_syncer_model': '选择负责唇形同步的模型',
		# uis
		'open_browser': '程序就绪时打开浏览器',
		'ui_layouts': '启动单个或多个UI布局 (选项: {choices}, ...)',
		'ui_workflow': '选择UI工作流程',
		# execution
		'execution_device_id': '指定用于处理的设备',
		'execution_providers': '使用不同提供程序加速模型推理 (选项: {choices}, ...)',
		'execution_thread_count': '指定处理时的并行线程数',
		'execution_queue_count': '指定每个线程处理的帧数',
		# memory
		'video_memory_strategy': '平衡快速处理和低显存使用',
		'system_memory_limit': '限制处理时可用的内存',
		# misc
		'skip_download': '跳过下载和远程查找',
		'log_level': '调整终端显示的消息严重程度',
		# run
		'run': '运行程序',
		'headless_run': '以无界面模式运行程序',
		'force_download': '强制自动下载并退出',
		# jobs
		'job_id': '指定任务ID',
		'job_status': '指定任务状态',
		'step_index': '指定步骤索引',
		# job manager
		'job_list': '按状态列出任务',
		'job_create': '创建草稿任务',
		'job_submit': '提交草稿任务成为队列任务',
		'job_submit_all': '提交所有草稿任务成为队列任务',
		'job_delete': '删除草稿、队列、失败或完成的任务',
		'job_delete_all': '删除所有草稿、队列、失败和完成的任务',
		'job_add_step': '向草稿任务添加步骤',
		'job_remix_step': '从草稿任务重混之前的步骤',
		'job_insert_step': '向草稿任务插入步骤',
		'job_remove_step': '从草稿任务移除步骤',
		# job runner
		'job_run': '运行队列任务',
		'job_run_all': '运行所有队列任务',
		'job_retry': '重试失败的任务',
		'job_retry_all': '重试所有失败的任务'
	},
	'uis':
	{
		'age_modifier_direction_slider': '年龄修改方向',
		'age_modifier_model_dropdown': '年龄修改模型',
		'apply_button': '应用',
		'benchmark_cycles_slider': '基准测试循环',
		'benchmark_runs_checkbox_group': '基准测试运行',
		'clear_button': '清除',
		'common_options_checkbox_group': '选项',
		'execution_providers_checkbox_group': '执行提供程序',
		'execution_queue_count_slider': '执行队列数量',
		'execution_thread_count_slider': '执行线程数量',
		'expression_restorer_factor_slider': '表情恢复因子',
		'expression_restorer_model_dropdown': '表情恢复模型',
		'face_debugger_items_checkbox_group': '人脸调试项目',
		'face_detector_angles_checkbox_group': '人脸检测角度',
		'face_detector_model_dropdown': '人脸检测模型',
		'face_detector_score_slider': '人脸检测分数',
		'face_detector_size_dropdown': '人脸检测大小',
		'face_editor_eyebrow_direction_slider': '眉毛方向',
		'face_editor_eye_gaze_horizontal_slider': '水平眼睛视线',
		'face_editor_eye_gaze_vertical_slider': '垂直眼睛视线',
		'face_editor_eye_open_ratio_slider': '眼睛开启比例',
		'face_editor_head_pitch_slider': '头部俯仰角度',
		'face_editor_head_roll_slider': '头部翻滚角度',
		'face_editor_head_yaw_slider': '头部偏航角度',
		'face_editor_lip_open_ratio_slider': '嘴唇开启比例',
		'face_editor_model_dropdown': '人脸编辑模型',
		'face_editor_mouth_grim_slider': '嘴巴严肃程度',
		'face_editor_mouth_position_horizontal_slider': '嘴巴水平位置',
		'face_editor_mouth_position_vertical_slider': '嘴巴垂直位置',
		'face_editor_mouth_pout_slider': '嘴巴撅起程度',
		'face_editor_mouth_purse_slider': '嘴巴抿起程度',
		'face_editor_mouth_smile_slider': '嘴巴微笑程度',
		'face_enhancer_blend_slider': '人脸增强混合',
		'face_enhancer_model_dropdown': '人脸增强模型',
		'face_landmarker_model_dropdown': '人脸特征点模型',
		'face_landmarker_score_slider': '人脸特征点分数',
		'face_mask_blur_slider': '人脸遮罩模糊',
		'face_mask_padding_bottom_slider': '人脸遮罩底部填充',
		'face_mask_padding_left_slider': '人脸遮罩左侧填充',
		'face_mask_padding_right_slider': '人脸遮罩右侧填充',
		'face_mask_padding_top_slider': '人脸遮罩顶部填充',
		'face_mask_regions_checkbox_group': '人脸遮罩区域',
		'face_mask_types_checkbox_group': '人脸遮罩类型',
		'face_selector_age_range_slider': '人脸选择器年龄范围',
		'face_selector_gender_dropdown': '人脸选择器性别',
		'face_selector_mode_dropdown': '人脸选择器模式',
		'face_selector_order_dropdown': '人脸选择器顺序',
		'face_selector_race_dropdown': '人脸选择器种族',
		'face_swapper_model_dropdown': '换脸模型',
		'face_swapper_pixel_boost_dropdown': '换脸像素提升',
		'frame_colorizer_blend_slider': '帧上色混合',
		'frame_colorizer_model_dropdown': '帧上色模型',
		'frame_colorizer_size_dropdown': '帧上色大小',
		'frame_enhancer_blend_slider': '帧增强混合',
		'frame_enhancer_model_dropdown': '帧增强模型',
		'job_list_status_checkbox_group': '任务状态',
		'job_manager_job_action_dropdown': '任务操作',
		'job_manager_job_id_dropdown': '任务ID',
		'job_manager_step_index_dropdown': '步骤索引',
		'job_runner_job_action_dropdown': '任务操作',
		'job_runner_job_id_dropdown': '任务ID',
		'lip_syncer_model_dropdown': '唇形同步模型',
		'log_level_dropdown': '日志级别',
		'output_audio_encoder_dropdown': '输出音频编码器',
		'output_image_or_video': '输出',
		'output_image_quality_slider': '输出图片质量',
		'output_image_resolution_dropdown': '输出图片分辨率',
		'output_path_textbox': '输出路径',
		'output_video_encoder_dropdown': '输出视频编码器',
		'output_video_fps_slider': '输出视频帧率',
		'output_video_preset_dropdown': '输出视频预设',
		'output_video_quality_slider': '输出视频质量',
		'output_video_resolution_dropdown': '输出视频分辨率',
		'preview_frame_slider': '预览帧',
		'preview_image': '预览',
		'processors_checkbox_group': '处理器',
		'reference_face_distance_slider': '参考人脸距离',
		'reference_face_gallery': '参考人脸',
		'refresh_button': '刷新',
		'source_file': '源文件',
		'start_button': '开始',
		'stop_button': '停止',
		'system_memory_limit_slider': '系统内存限制',
		'target_file': '目标文件',
		'temp_frame_format_dropdown': '临时帧格式',
		'terminal_textbox': '终端',
		'trim_frame_slider': '裁剪帧',
		'ui_workflow': 'UI工作流程',
		'video_memory_strategy_dropdown': '显存策略',
		'webcam_fps_slider': '网络摄像头帧率',
		'webcam_image': '网络摄像头',
		'webcam_mode_radio': '网络摄像头模式',
		'webcam_resolution_dropdown': '网络摄像头分辨率'
	}
}


def get(key : str) -> Optional[str]:
	if '.' in key:
		section, name = key.split('.')
		if section in WORDING and name in WORDING.get(section):
			return WORDING.get(section).get(name)
	if key in WORDING:
		return WORDING.get(key)
	return None
