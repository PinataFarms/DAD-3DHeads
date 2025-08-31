import json
from tqdm import tqdm
from fire import Fire
from utils import read_img


def generate_gt(
        base_path: str,
        subset_name: str ='val',
        with_attributes: bool = False
):
    assert not (subset_name == 'val' and with_attributes), f"Attributes not supported for subset '{subset_name}'"
    with open(f'{base_path}/DAD-3DHeadsDataset/{subset_name}/{subset_name}.json', "r") as f:
        subset_anno = json.load(f)

    subset_json = []
    for el in tqdm(subset_anno):
        annotation_path = f'{base_path}/DAD-3DHeadsDataset/{subset_name}/annotations/{el["item_id"]}.json'
        image = read_img(f'{base_path}/DAD-3DHeadsDataset/{subset_name}/images/{el["item_id"]}.png')
        image_height = image.shape[0]
        anno = json.loads(open(annotation_path).read())
        el_dict = {
                'id': el['item_id'],
                'bbox': el['bbox'],
                'vertices': anno['vertices'],
                'model_view_matrix': anno['model_view_matrix'],
                'projection_matrix': anno['projection_matrix'],
                'image_height': image_height
        }
        if with_attributes:
            el_dict['attributes'] = el['attributes']
        subset_json.append(el_dict)

    suffix = '_with_attributes' if with_attributes else ''
    output_filename = f'data/ground_truth_{subset_name}{suffix}.json'
    with open(output_filename, "w") as out:
        json.dump(subset_json, out)


if __name__ == "__main__":
    Fire(generate_gt)