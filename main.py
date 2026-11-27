"""Command-line entry point for the vision toolkit."""

import argparse
import sys
from importlib import import_module


TOOL_GROUPS = {
    'augment': ('影像擴增與旋轉', {
        'annotated': ('image_augmentation_script', 'VOC、CVAT、YOLO 水平框同步擴增'),
        'obb': ('obb_image_augmentation_script', 'YOLO OBB 四角點同步擴增'),
        'image-only': ('image_augmentation_script', '無標註的影像擴增'),
        'rotate': ('image_rotator', '批次順時針旋轉 90°、180°、270°'),
        'voc-balance': ('voc_balancer', '重採樣並平衡既有 VOC 資料集'),
    }),
    'crop': ('影像裁切與座標選取', {
        'center': ('image_center_crop', '依比例裁切中心區域'),
        'coordinates': ('crop_xyxy', '依指定 XYXY 座標裁切'),
        'voc': ('crop_voc_objects', '依 VOC 物件框裁切'),
        'pick-points': ('click_xy', '在影像上選取座標'),
    }),
    'dataset': ('資料集整理', {
        'split': ('image_dataset_splitter', '分割影像及其配對標註'),
        'sample': ('weighted_image_sampler', '按子資料夾的影像占比抽樣'),
        'filter': ('filter_dataset', '依 VOC 或 CVAT 標註篩選資料'),
        'count-voc': ('voc_label_counter', '統計 VOC 類別與影像配對'),
        'merge-voc': ('merge_cvat_voc', '合併並重新命名 VOC 資料'),
        'organize': ('image_organizer', '依產線影像檔名與資料夾整理'),
    }),
    'convert': ('標註格式轉換與打包', {
        'voc-to-yolo': ('voc2yolo', '將 VOC 水平框轉成 YOLO 資料集'),
        'cvat-to-yolo-seg': ('cvat2yolo_seg', '將 CVAT 多邊形轉成 YOLO 分割資料集'),
        'prepare-obb': ('obb2yolo', '修正越界 OBB 並產出 YOLO 資料集'),
        'yolo-to-cvat': ('yolo_to_cvat_converter', '打包 YOLO 資料以匯入 CVAT'),
        'mask-to-cvat': ('mask2polygon', '將遮罩轉為 CVAT 多邊形'),
        'colorize-mask': ('process_masks', '批次處理及上色遮罩'),
    }),
    'inspect': ('影像檢測與標註', {
        'quality': ('image_brightness_detector', '評估亮度、對比度與影像品質'),
        'ocr': ('aoi_ocr_classifier', '以 OCR 判斷 AOI 影像方向'),
        'fov': ('fov_report', '產線 FOV 分類驗證與報表'),
        'annotate': ('image_annotator', '開啟人工框選工具'),
    }),
    'model': ('模型推論、匯出與模組建置', {
        'predict-voc': ('yolo_pascal_voc_converter', '以 YOLO 推論產生 VOC 標註'),
        'export-tflite': ('pt2tflite', '匯出量化 TFLite 模型'),
        'create-pol-module': ('create_pol_module', '建立極性檢測模組'),
    }),
    'report': ('輔助輸出', {
        'grid': ('make_grid', '將多張影像排成網格'),
        'json-to-excel': ('json_to_excel', '將元件 JSON 清單匯出成 Excel'),
    }),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description='AI 視覺開發工具箱',
        epilog='使用「python main.py <群組> <工具> --help」查看個別工具的參數。',
    )
    groups = parser.add_subparsers(dest='group', metavar='群組')
    for name, (description, commands) in TOOL_GROUPS.items():
        group = groups.add_parser(name, help=description, description=description)
        modes = group.add_subparsers(dest='mode', metavar='工具')
        for mode, (module, summary) in commands.items():
            command = modes.add_parser(mode, help=summary, add_help=False)
            command.set_defaults(module=module)
        group.set_defaults(show_help=group.print_help)

    args = parser.parse_args(argv[:2])
    if args.group is None:
        parser.print_help()
        return 0
    if args.mode is None:
        args.show_help()
        return 0

    options = argv[2:]
    if (args.group, args.mode) == ('inspect', 'annotate') and options in (['-h'], ['--help']):
        print('使用 python main.py inspect annotate 開啟框選介面。需要 PyQt5 與桌面環境。')
        return 0
    tool = import_module(f'vision_toolkit.{args.module}')
    if (args.group, args.mode) == ('augment', 'image-only'):
        return tool.main(options, image_only=True) or 0
    return tool.main(options) or 0


if __name__ == '__main__':
    raise SystemExit(main())
