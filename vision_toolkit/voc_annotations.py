"""Shared label extraction for Pascal VOC XML."""

import xml.etree.ElementTree as ET


def object_labels(root: ET.Element, *, strip: bool = True) -> list[str]:
    """Read direct object names, keeping their order and repeated instances."""
    labels = []
    for obj in root.findall('object'):
        label = obj.findtext('name', default='')
        if strip:
            label = label.strip()
        if label:
            labels.append(label)
    return labels
