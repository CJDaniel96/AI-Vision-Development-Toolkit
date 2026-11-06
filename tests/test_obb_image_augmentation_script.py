import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from obb_image_augmentation_script import (
    augment_obb,
    build_pipeline,
    load_yolo_obb_annotations,
    main,
    run_augmentation,
    save_yolo_obb_annotations,
)


BOXES = [
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5], [0.1, 0.5]],
]
# 在 200×80 影像上，第一個框為旋轉 45° 的正方形，四邊皆超出影像。
OUTSIDE_BOXES = [
    [[-0.05, 0.5], [0.5, -0.875], [1.05, 0.5], [0.5, 1.875]],
    BOXES[1],
]


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'sample.txt'

    def test_round_trip_multiple_classes_and_boundary_coordinates(self):
        save_yolo_obb_annotations(self.path, BOXES, [7, 19])
        boxes, labels = load_yolo_obb_annotations(self.path)
        np.testing.assert_allclose(boxes, BOXES)
        self.assertEqual(labels, [7, 19])
        self.assertTrue(all(len(line.split()) == 9 for line in self.path.read_text().splitlines()))

    def test_empty_label_is_background_but_missing_label_is_error(self):
        with self.assertRaises(FileNotFoundError):
            load_yolo_obb_annotations(self.path)
        self.path.write_text('\n  \n')
        self.assertEqual(load_yolo_obb_annotations(self.path), ([], []))

    def test_cvat_out_of_bounds_coordinates_are_loaded_without_clipping(self):
        save_yolo_obb_annotations(self.path, OUTSIDE_BOXES, [7, 19])
        boxes, labels = load_yolo_obb_annotations(self.path)
        np.testing.assert_allclose(boxes, OUTSIDE_BOXES)
        self.assertEqual(labels, [7, 19])

    def test_invalid_labels_report_filename_and_line(self):
        for line in (
            '0 0.5 0.5 0.2 0.2',  # HBB 不能當 OBB 讀入
            '-1 0 0 1 0 1 1 0 1',
            '1.5 0 0 1 0 1 1 0 1',
            '0 nan 0 1 0 1 1 0 1',
            '0 inf 0 1 0 1 1 0 1',
            '0 0 0 0 0 0 0 0 0',
            '0 0 0 1 1 1 0 0 1',  # 自交四邊形
        ):
            with self.subTest(line=line):
                self.path.write_text('\n' + line)
                with self.assertRaisesRegex(ValueError, 'sample.txt:2'):
                    load_yolo_obb_annotations(self.path)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).integers(0, 256, (80, 200, 3), dtype=np.uint8)

    def test_padding_preserves_source_pixels_and_every_box(self):
        image, boxes, labels = augment_obb(
            self.image, OUTSIDE_BOXES, [7, 19], build_pipeline('original'),
        )
        self.assertEqual(image.shape, (220, 220, 3))
        np.testing.assert_array_equal(image[70:150, 10:210], self.image)
        background = np.ones((220, 220), dtype=bool)
        background[70:150, 10:210] = False
        self.assertTrue(np.all(image[background] == 114))
        np.testing.assert_allclose(
            np.asarray(boxes) * [220, 220],
            np.asarray(OUTSIDE_BOXES) * [200, 80] + [10, 70],
            atol=1e-5,
        )
        self.assertEqual(labels, [7, 19])
        self.assertTrue(np.all(np.asarray(boxes) >= 0))
        self.assertTrue(np.all(np.asarray(boxes) <= 1))

    def test_fractional_pixel_overflow_rounds_padding_outward(self):
        points = np.array([
            [[-2.5, 10], [10, -2.5], [22.5, 10], [10, 22.5]],
            [[187.5, 70], [200, 57.5], [212.5, 70], [200, 82.5]],
        ])
        image, boxes, labels = augment_obb(
            self.image, (points / [200, 80]).tolist(), [7, 19],
            build_pipeline('original'),
        )
        self.assertEqual(image.shape, (86, 216, 3))
        np.testing.assert_array_equal(image[3:83, 3:203], self.image)
        np.testing.assert_allclose(np.asarray(boxes) * [216, 86], points + [3, 3], atol=1e-5)
        self.assertEqual(labels, [7, 19])

    def test_padded_boxes_keep_pixel_geometry_through_flips_and_rotation(self):
        source_points = (np.asarray(OUTSIDE_BOXES) * [200, 80]).reshape(-1, 2)
        source_distances = np.linalg.norm(source_points[:, None] - source_points[None, :], axis=-1)
        for profile, angle in (
            ('horizontal_flip', None), ('vertical_flip', None),
            ('rotation', 90), ('rotation', 33),
        ):
            with self.subTest(profile=profile, angle=angle):
                image, boxes, labels = augment_obb(
                    self.image, OUTSIDE_BOXES, [7, 19], build_pipeline(profile, angle=angle),
                )
                points = (np.asarray(boxes) * [image.shape[1], image.shape[0]]).reshape(-1, 2)
                distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
                np.testing.assert_allclose(distances, source_distances, atol=1e-4)
                self.assertTrue(np.all(np.asarray(boxes) >= 0))
                self.assertTrue(np.all(np.asarray(boxes) <= 1))
                self.assertEqual(labels, [7, 19])

    def test_original_mode_keeps_in_bounds_images_and_backgrounds_unchanged(self):
        for boxes, labels in ((BOXES, [7, 19]), ([], [])):
            with self.subTest(labels=labels):
                image, result, result_labels = augment_obb(
                    self.image, boxes, labels, build_pipeline('original'),
                )
                np.testing.assert_array_equal(image, self.image)
                np.testing.assert_allclose(result, boxes, atol=1e-7)
                self.assertEqual(result_labels, labels)

    def test_flips_use_normalized_edges_without_pixel_offset(self):
        for profile, axis in (('horizontal_flip', 1), ('vertical_flip', 0)):
            with self.subTest(profile=profile):
                image, boxes, labels = augment_obb(
                    self.image, BOXES, [7, 19], build_pipeline(profile),
                )
                expected = np.array(BOXES, dtype=float)
                coordinate = 0 if axis == 1 else 1
                expected[:, :, coordinate] = 1 - expected[:, :, coordinate]
                np.testing.assert_allclose(boxes, expected, atol=1e-7)
                np.testing.assert_array_equal(image, np.flip(self.image, axis=axis))
                self.assertEqual(labels, [7, 19])

    def test_right_angle_rotations_match_image_and_preserve_all_corners(self):
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                image, boxes, labels = augment_obb(
                    self.image, BOXES, [7, 19], build_pipeline('rotation', angle=angle),
                )
                np.testing.assert_array_equal(image, np.rot90(self.image, angle // 90))
                expected = np.array(BOXES, dtype=float)
                for _ in range(angle // 90):
                    expected = np.stack([expected[:, :, 1], 1 - expected[:, :, 0]], axis=-1)
                np.testing.assert_allclose(boxes, expected, atol=1e-7)
                self.assertEqual(labels, [7, 19])

    def test_arbitrary_rotation_preserves_pixel_geometry_and_alignment(self):
        image = np.zeros((80, 200, 3), dtype=np.uint8)
        image[16:40, 20:60] = 255
        for angle in (33, 45, 137, 315):
            with self.subTest(angle=angle):
                output, boxes, labels = augment_obb(
                    image, BOXES, [7, 19], build_pipeline('rotation', angle=angle),
                )
                height, width = output.shape[:2]
                points = np.asarray(boxes) * [width, height]
                self.assertTrue(np.all(np.asarray(boxes) >= 0))
                self.assertTrue(np.all(np.asarray(boxes) <= 1))
                self.assertAlmostEqual(np.linalg.norm(points[1, 1] - points[1, 0]), 40, places=4)
                self.assertAlmostEqual(np.linalg.norm(points[1, 2] - points[1, 1]), 24, places=4)
                self.assertAlmostEqual(
                    np.dot(points[1, 1] - points[1, 0], points[1, 2] - points[1, 1]),
                    0, places=2,
                )
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.rint(points[1] - 0.5).astype(np.int32), 1)
                actual = output[:, :, 0] > 200
                iou = np.count_nonzero(actual & (mask != 0)) / np.count_nonzero(actual | (mask != 0))
                self.assertGreater(iou, 0.9)
                self.assertEqual(labels, [7, 19])

    def test_photometric_augmentation_keeps_annotations(self):
        for profile in ('brightness_contrast', 'hue_saturation'):
            with self.subTest(profile=profile):
                _, boxes, labels = augment_obb(self.image, BOXES, [7, 19], build_pipeline(profile))
                np.testing.assert_allclose(boxes, BOXES, atol=1e-7)
                self.assertEqual(labels, [7, 19])

    def test_empty_background_survives_mixed_augmentation(self):
        image, boxes, labels = augment_obb(self.image, [], [], build_pipeline('mixed'))
        self.assertGreater(image.size, 0)
        self.assertEqual((boxes, labels), ([], []))

    def test_random_affine_keeps_border_boxes_on_small_and_narrow_images(self):
        for shape in ((2, 3, 3), (17, 301, 3)):
            image = np.zeros(shape, dtype=np.uint8)
            for profile in ('mixed', 'rotation'):
                for seed in (8, 11, 19, 20, 25, 29, 30):
                    with self.subTest(shape=shape, profile=profile, seed=seed):
                        _, boxes, labels = augment_obb(
                            image, BOXES[:1], [7], build_pipeline(profile, seed=seed),
                        )
                        self.assertEqual(labels, [7])
                        self.assertTrue(np.all(np.asarray(boxes) >= 0))
                        self.assertTrue(np.all(np.asarray(boxes) <= 1))

    def test_seed_reproduces_sequence_while_versions_vary(self):
        first, second = build_pipeline('mixed', seed=12), build_pipeline('mixed', seed=12)
        generated = []
        for _ in range(4):
            result = augment_obb(self.image, BOXES, [7, 19], first)
            repeated = augment_obb(self.image, BOXES, [7, 19], second)
            np.testing.assert_array_equal(result[0], repeated[0])
            np.testing.assert_allclose(result[1], repeated[1])
            generated.append(result[0].tobytes())
        self.assertGreater(len(set(generated)), 1)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / 'images'
        self.labels = self.root / 'labels'
        self.output = self.root / 'output'
        self.images.mkdir()
        self.labels.mkdir()

    def write_sample(self, name='sample', boxes=BOXES, labels=(7, 19)):
        image_path = self.images / f'{name}.png'
        label_path = self.labels / f'{name}.txt'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), np.full((80, 200, 3), 150, dtype=np.uint8))
        save_yolo_obb_annotations(label_path, boxes, list(labels))

    def run_dataset(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_augmentation(
                self.images, self.labels, self.output,
                kwargs.pop('augmentation_types', ['horizontal_flip']), **kwargs,
            )

    def test_each_image_and_profile_gets_requested_version_count(self):
        self.write_sample()
        self.write_sample('background', [], [])
        summary = self.run_dataset(
            augmentation_types=['horizontal_flip', 'brightness_contrast'], num_augmentations=3,
        )
        self.assertEqual((summary.generated, summary.skipped_sources, summary.failed_outputs), (12, 0, 0))
        self.assertEqual(len(list((self.output / 'images').glob('*.png'))), 12)
        self.assertEqual(len(list((self.output / 'labels').glob('*.txt'))), 12)
        self.assertEqual((self.output / 'labels/background_horizontal_flip_1.txt').read_text(), '')
        _, labels = load_yolo_obb_annotations(self.output / 'labels/sample_horizontal_flip_1.txt')
        self.assertEqual(labels, [7, 19])

    def test_fixed_steps_cover_all_angles_below_360(self):
        self.write_sample()
        summary = self.run_dataset(augmentation_types=['rotation'], rotate_step=100)
        self.assertEqual(summary.generated, 4)
        self.assertEqual(
            {path.stem for path in (self.output / 'labels').glob('*.txt')},
            {f'sample_rot{angle}_1' for angle in (0, 100, 200, 300)},
        )

    def test_recursive_preserves_split_paths_with_same_filenames(self):
        self.write_sample('train/sample')
        self.write_sample('val/sample')
        summary = self.run_dataset(recursive=True)
        self.assertEqual(summary.generated, 2)
        for split in ('train', 'val'):
            self.assertTrue((self.output / 'labels' / split / 'sample_horizontal_flip_1.txt').exists())

    def test_invalid_and_corrupt_sources_are_skipped_but_missing_labels_are_augmented(self):
        for name in ('good', 'invalid', 'missing', 'corrupt'):
            self.write_sample(name)
        (self.labels / 'invalid.txt').write_text('0 0.5 0.5 0.2 0.2')
        (self.labels / 'missing.txt').unlink()
        (self.images / 'corrupt.png').write_bytes(b'broken image')
        summary = self.run_dataset()
        self.assertEqual((summary.generated, summary.skipped_sources, summary.failed_outputs), (2, 2, 0))
        self.assertEqual((self.output / 'labels/missing_horizontal_flip_1.txt').read_text(), '')

    def test_unlabeled_backgrounds_get_all_profiles_and_requested_versions(self):
        self.write_sample('train/background', [], [])
        source_label = self.labels / 'train/background.txt'
        source_label.unlink()
        source_label.parent.rmdir()
        image = np.random.default_rng(0).integers(0, 256, (80, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(self.images / 'train/background.png'), image)
        profiles = [
            'original', 'horizontal_flip', 'vertical_flip', 'brightness_contrast',
            'hue_saturation', 'mixed', 'rotation',
        ]
        summary = self.run_dataset(
            augmentation_types=profiles, num_augmentations=2, recursive=True,
        )
        self.assertEqual((summary.generated, summary.skipped_sources, summary.failed_outputs), (14, 0, 0))
        for profile in profiles:
            for index in (1, 2):
                stem = f'background_{profile}_{index}'
                label_path = self.output / 'labels/train' / f'{stem}.txt'
                image_path = self.output / 'images/train' / f'{stem}.png'
                self.assertEqual(label_path.read_text(), '')
                self.assertIsNotNone(cv2.imread(str(image_path)))
        flipped = cv2.imread(str(self.output / 'images/train/background_horizontal_flip_1.png'))
        np.testing.assert_array_equal(flipped, np.flip(image, axis=1))
        self.assertFalse(source_label.exists())

    def test_cli_rotates_images_without_label_files_at_every_fixed_angle(self):
        self.write_sample('background', [], [])
        (self.labels / 'background.txt').unlink()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main([
                '-i', str(self.images), '-o', str(self.output),
                '-a', 'rotation', '--rotate_step', '45',
            ])
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.output / 'images').glob('*.png'))), 8)
        labels = list((self.output / 'labels').glob('*.txt'))
        self.assertEqual({path.stem for path in labels}, {
            f'background_rot{angle}_1' for angle in range(0, 360, 45)
        })
        self.assertTrue(all(path.read_text() == '' for path in labels))

    def test_unreadable_label_is_skipped_instead_of_treated_as_background(self):
        self.write_sample()
        with patch('obb_image_augmentation_script.load_yolo_obb_annotations', side_effect=PermissionError('unreadable')):
            summary = self.run_dataset()
        self.assertEqual((summary.generated, summary.skipped_sources), (0, 1))
        self.assertFalse(self.output.exists())

    def test_existing_outputs_are_preserved(self):
        self.write_sample()
        self.run_dataset()
        path = self.output / 'labels/sample_horizontal_flip_1.txt'
        original = path.read_bytes()
        summary = self.run_dataset()
        self.assertEqual((summary.generated, summary.failed_outputs), (0, 1))
        self.assertEqual(path.read_bytes(), original)

    def test_image_write_failure_does_not_create_annotation(self):
        self.write_sample()
        with patch('obb_image_augmentation_script.cv2.imwrite', return_value=False):
            summary = self.run_dataset()
        self.assertEqual((summary.generated, summary.failed_outputs), (0, 1))
        self.assertEqual(list((self.output / 'labels').iterdir()), [])

    def test_label_write_failure_removes_written_image(self):
        self.write_sample()
        with patch('obb_image_augmentation_script.save_yolo_obb_annotations', side_effect=OSError('disk full')):
            summary = self.run_dataset()
        self.assertEqual((summary.generated, summary.failed_outputs), (0, 1))
        self.assertEqual(list((self.output / 'images').iterdir()), [])

    def test_rejects_duplicate_stems_before_writing(self):
        self.write_sample()
        cv2.imwrite(str(self.images / 'sample.jpg'), np.zeros((80, 200, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, '重複影像檔名'):
            self.run_dataset()
        self.assertFalse(self.output.exists())

    def test_cli_supports_colocated_labels(self):
        self.write_sample()
        (self.labels / 'sample.txt').replace(self.images / 'sample.txt')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(['-i', str(self.images), '-o', str(self.output), '-a', 'rotation', '--rotate_step', '45'])
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.output / 'labels').glob('*.txt'))), 8)

    def test_cli_repairs_cvat_labels_without_other_augmentation(self):
        self.write_sample(boxes=OUTSIDE_BOXES)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main([
                '-i', str(self.images), '-l', str(self.labels), '-o', str(self.output),
                '-a', 'original',
            ])
        self.assertEqual(code, 0)
        image = cv2.imread(str(self.output / 'images/sample_original_1.png'))
        boxes, labels = load_yolo_obb_annotations(self.output / 'labels/sample_original_1.txt')
        self.assertEqual(image.shape, (220, 220, 3))
        np.testing.assert_allclose(
            np.asarray(boxes) * [220, 220],
            np.asarray(OUTSIDE_BOXES) * [200, 80] + [10, 70],
            atol=2e-6,
        )
        self.assertEqual(labels, [7, 19])
        self.assertTrue(np.all(image[70:150, 10:210] == 150))

    def test_invalid_cli_options_exit_before_writing(self):
        base = ['-i', str(self.images), '-o', str(self.output)]
        for options in (
            ['-n', '0'], ['--rotate_step', '0'], ['--rotate_limit', 'nan'],
            ['--rotate_limit', '181'], ['--rotate_step', '360'],
            ['--rotate_step', '45'],
            ['-a', 'rotation', '--rotate_step', '45', '-n', '2'],
            ['-a', 'rotation', '--rotate_step', '45', '--rotate_limit', '10'],
        ):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(base + options)
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
