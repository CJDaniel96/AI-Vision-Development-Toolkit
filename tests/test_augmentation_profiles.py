import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision_toolkit.augmentation_profiles import load_profiles
from vision_toolkit.image_augmentation_script import detect_and_load_annotations


TOOLKIT = Path(__file__).resolve().parents[1] / 'main.py'


class AugmentationProfilesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / 'profiles.json'

    def test_shared_recipe_works_in_all_three_cli_modes(self):
        self.config.write_text(json.dumps({'shared': {
            'test_bright': [{
                'transform': 'RandomBrightnessContrast',
                'brightness_limit': [0.2, 0.2],
                'contrast_limit': 0,
                'p': 1,
            }],
        }}))
        profiles = load_profiles(self.config)
        for mode, default in (('image_only', 'default'), ('annotated', 'hz_pde'), ('obb', 'mixed')):
            self.assertIn(default, profiles[mode])

        for mode in ('annotated', 'image-only', 'obb'):
            with self.subTest(mode=mode):
                source, output = self.root / mode, self.root / f'{mode}-out'
                source.mkdir()
                cv2.imwrite(str(source / 'sample.png'), np.zeros((30, 40, 3), dtype=np.uint8))
                if mode == 'annotated':
                    (source / 'sample.txt').write_text('0 0.5 0.5 0.5 0.5\n')
                if mode == 'obb':
                    (source / 'sample.txt').write_text('0 .25 .25 .75 .25 .75 .75 .25 .75\n')
                selection = '--image_only_profile' if mode == 'image-only' else '-a'
                completed = subprocess.run(
                    [sys.executable, str(TOOLKIT), 'augment', mode,
                     '-i', str(source), '-o', str(output),
                     '--config', str(self.config), selection, 'test_bright'],
                    cwd=self.root, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                images = list(output.rglob('*.png'))
                self.assertEqual(len(images), 1)
                self.assertTrue(np.all(cv2.imread(str(images[0])) == 51))
                if mode != 'image-only':
                    fields = next((output / 'labels').glob('*.txt')).read_text().split()
                    self.assertEqual(len(fields), 5 if mode == 'annotated' else 9)

    def test_overriding_connector_replaces_the_entire_recipe(self):
        recipe = [{'transform': 'RandomGamma', 'gamma_limit': [98, 102], 'p': 1}]
        self.config.write_text(json.dumps({'shared': {'connector': recipe}}))
        profiles = load_profiles(self.config)
        for mode in profiles:
            self.assertEqual(profiles[mode]['connector'], recipe)

    def test_invalid_recipes_fail_validation(self):
        for mode, spec in (
            ('obb', {'transform': 'Affine'}),
            ('shared', {'transform': 'HorizontalFlip'}),
            ('annotated', {'transform': 'RandomGamma', 'typo': 1}),
            ('image_only', {'transform': 'OneOf', 'p': 1.5, 'transforms': []}),
        ):
            with self.subTest(mode=mode):
                self.config.write_text(json.dumps({mode: {'example': [spec]}}))
                with self.assertRaisesRegex(ValueError, rf'{mode}\.example'):
                    load_profiles(self.config)

    def test_annotated_mode_skips_obb_instead_of_reinterpreting_coordinates(self):
        label = self.root / 'obb.txt'
        label.write_text('0 .25 .25 .75 .25 .75 .75 .25 .75\n')
        loaded = detect_and_load_annotations(label.with_suffix('.png'), self.root)
        self.assertEqual(loaded, (None, None, None, None, None))


if __name__ == '__main__':
    unittest.main()
