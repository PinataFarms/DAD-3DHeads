import json
from tqdm import tqdm
from fire import Fire
from utils import read_img


def generate_gt(
        base_path: str,
        output_filename: str = 'data/ground_truth_val.json',
):
    with open(f'{base_path}/DAD-3DHeadsDataset/val/val.json', "r") as f:
        val_anno = json.load(f)

    val_json = []
    for el in tqdm(val_anno):
        annotation_path = f'{base_path}/DAD-3DHeadsDataset/val/annotations/{el["item_id"]}.json'
        image = read_img(f'{base_path}/DAD-3DHeadsDataset/val/images/{el["item_id"]}.png')
        image_height = image.shape[0]
        anno = json.loads(open(annotation_path).read())
        val_json.append(
            {
                'id': el['item_id'],
                'bbox': el['bbox'],
                'vertices': anno['vertices'],
                'model_view_matrix': anno['model_view_matrix'],
                'projection_matrix': anno['projection_matrix'],
                'image_height': image_height
            }
        )

    with open(output_filename, "w") as out:
        json.dump(val_json, out)


if __name__ == "__main__":
    Fire(generate_gt)