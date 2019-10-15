import pickle

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

dataset= '/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/train.pickle'

path = []
boxes = []
points = []
eyes = []
with open(dataset, 'rb') as f:
    data = pickle.load(f)
    num_data = len(data)

    for i in range(5):
        # print(data[i]['filename'])
        # print(data[i]['ann']['bboxes'][-1,:])
        # print(data[i]['gaze_cx'])
        # print(data[i]['gaze_cy'])

        path.append(data[i]['filename'])
        boxes.append(data[i]['ann']['bboxes'][-1,:])
        points.append([data[i]['gaze_cx'],data[i]['gaze_cy']])
        eyes.append([data[i]['hx'],data[i]['hy']])

boxes = totuple(boxes)
print(path)
print(boxes)
print(points)
print(eyes)
