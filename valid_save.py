
import torch

from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

from tqdm import tqdm
from args import get_parser  # fcl add
parser = get_parser()  # fcl add
args = parser.parse_args()


def evaluate_save(local_rank, model, loader, num_classes):
    # save eval img

    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    total_pred = 0

    with torch.no_grad():
        for img, mask, sentences, attentions in tqdm(loader):
            img = img.cuda(local_rank)
            sentences, attentions = sentences.cuda(local_rank), attentions.cuda(local_rank)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            total_pred += img.shape[0]

            pred, _ = model(img, sentences, l_mask=attentions)
            pred = pred.argmax(dim=1)

            intersection1, union1, target1 = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), num_classes, 255)


            reduced_intersection1 = torch.from_numpy(intersection1).cuda(local_rank)
            reduced_union1 = torch.from_numpy(union1).cuda(local_rank)


            intersection_meter.update(reduced_intersection1.cpu().numpy())
            union_meter.update(reduced_union1.cpu().numpy())

    iou_class1 = intersection_meter.sum / (union_meter.sum + 1e-10)
    # mIOU1 = np.mean(iou_class1) * 100.0   # 每个类别的平均IOU
    oIOU1 = iou_class1[1] * 100.0

    result = {}

    result['IOU'] = oIOU1
    # result['iou_class'] = iou_class1

    return result


if __name__ == '__main__':
    evaluate_save()
