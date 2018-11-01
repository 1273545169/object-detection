

input_shape = 416  # width=height # 608 or 416 or 320
grid_shapes = [13, 26, 52]

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

num_class = 80


def load_coco_names(file_name):
    classes = []
    with open(file_name) as f:
        for name in f.readlines():
            classes.append(name.strip())
    return classes





