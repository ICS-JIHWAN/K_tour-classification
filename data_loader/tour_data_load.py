import cv2
import torch

class tour_data_load():
    def __init__(self, img_path_list, text_vectors, label_list, transforms, train=False):
        self.img_path_list = img_path_list
        self.text_vectors = text_vectors
        self.label_list = label_list
        self.transforms = transforms
        self.train = train
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        return_dict = dict()

        # NLP
        text_vector = self.text_vectors[index]

        # Image
        img_path = '/storage/jhchoi/tour/open' + self.img_path_list[index][1:]
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        # Label
        if self.train:
            label = self.label_list[index]
            return_dict['img'] = torch.Tensor(image)
            return_dict['text'] = torch.Tensor(text_vector).view(-1)
            return_dict['label'] = torch.tensor(label, dtype=torch.float32)
            return return_dict
        else:
            return_dict['img'] = torch.Tensor(image)
            return_dict['text'] = torch.Tensor(text_vector).view(-1)
            return return_dict