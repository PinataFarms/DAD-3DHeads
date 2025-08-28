import json
from tqdm import tqdm
from fire import Fire
from utils import read_img


def generate_gt(
        base_path: str,
        subset_name: str ='val'
):
    with open(f'{base_path}/DAD-3DHeadsDataset/{subset_name}/{subset_name}.json', "r") as f:
        subset_anno = json.load(f)

    subset_json = []
    for el in tqdm(subset_anno):
        annotation_path = f'{base_path}/DAD-3DHeadsDataset/{subset_name}/annotations/{el["item_id"]}.json'
        image = read_img(f'{base_path}/DAD-3DHeadsDataset/{subset_name}/images/{el["item_id"]}.png')
        image_height = image.shape[0]
        anno = json.loads(open(annotation_path).read())
        subset_json.append(
            {
                'id': el['item_id'],
                'bbox': el['bbox'],
                'vertices': anno['vertices'],
                'model_view_matrix': anno['model_view_matrix'],
                'projection_matrix': anno['projection_matrix'],
                'image_height': image_height
            }
        )

    output_filename = f'data/ground_truth_{subset_name}.json'
    with open(output_filename, "w") as out:
        json.dump(subset_json, out)


if __name__ == "__main__":
    Fire(generate_gt)