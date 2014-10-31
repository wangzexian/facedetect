#!/usr/bin/env python
# facedetect: a simple face detector for batch processing
# Copyright(c) 2013 by wave++ "Yuri D'Elia" <wavexx@thregr.org>
# Distributed under GPL2 (see COPYING) WITHOUT ANY WARRANTY.
from __future__ import print_function

import argparse
import cv2
import numpy as np
import math
import sys
import os


# CV compatibility stubs
if 'IMREAD_GRAYSCALE' not in dir(cv2):
    cv2.IMREAD_GRAYSCALE = 0L


# Profiles
PROFILES = {
    'HAAR_FRONTALFACE_ALT2': 'haarcascade_frontalface_alt2.xml'
}


# Support functions
def error(msg):
    sys.stderr.write("{}: error: {}\n".format(os.path.basename(sys.argv[0]), msg))


def fatal(msg):
    error(msg)
    sys.exit(1)


def check_profiles():
    for k, v in PROFILES.iteritems():
        if not os.path.exists(v):
            fatal("cannot load {} from {}".format(k, v))


def rank(im, rects):
    scores = []
    best = None

    for i in range(len(rects)):
        rect = rects[i]
        b = min(rect[2], rect[3]) / 10.
        rx = (rect[0] + b, rect[0] + rect[2] - b)
        ry = (rect[1] + b, rect[1] + rect[3] - b)
        roi = im[ry[0]:ry[1], rx[0]:rx[1]]
        s = (rect[2] + rect[3]) / 2.

        scale = 100. / max(rect[2], rect[3])
        dsize = (int(rect[2] * scale), int(rect[3] * scale))
        roi_n = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)
        roi_l = cv2.Laplacian(roi_n, cv2.CV_8U)
        e = np.sum(roi_l) / (dsize[0] * dsize[1])

        dx = im.shape[1] / 2 - rect[0] + rect[2] / 2
        dy = im.shape[0] / 2 - rect[1] + rect[3] / 2
        d = math.sqrt(dx ** 2 + dy ** 2) / (max(im.shape) / 2)

        scores.append({'s': s, 'e': e, 'd': d})

    sMax = max([x['s'] for x in scores])
    eMax = max([x['e'] for x in scores])

    for i in range(len(scores)):
        s = scores[i]
        sN = s['sN'] = s['s'] / sMax
        eN = s['eN'] = s['e'] / eMax
        f = s['f'] = eN * 0.7 + (1 - s['d']) * 0.1 + sN * 0.2

    ranks = range(len(scores))
    ranks = sorted(ranks, reverse=True, key=lambda x: scores[x]['f'])
    for i in range(len(scores)):
        scores[ranks[i]]['RANK'] = i

    return scores, ranks[0]


def __main__():
    ap = argparse.ArgumentParser(description='A simple face detector for batch processing')
    ap.add_argument('--biggest', action="store_true",
                    help='Extract only the biggest face')
    ap.add_argument('--best', action="store_true",
                    help='Extract only the best matching face')
    ap.add_argument('--crop', action="store_true",
                    help='Crop face. Requires --biggest or --best and -o.')
    ap.add_argument('--resize', type=int,
                    help='Resize output image to this size. Requires --crop.')
    ap.add_argument('--scale', type=float, default=1,
                    help='Enlarge/reduce detected face. Requires --crop.')
    ap.add_argument('--grayscale', action="store_true",
                    help='Convert image to grayscale. Requires --crop.')
    ap.add_argument('-c', '--center', action="store_true",
                    help='Print only the center coordinates')
    ap.add_argument('-q', '--query', action="store_true",
                    help='Query only (exit 0: face detected, 2: no detection)')
    ap.add_argument('-o', '--output', help='Image output file')
    ap.add_argument('-d', '--debug', action="store_true",
                    help='Add debugging metrics in the image output file')
    ap.add_argument('file', help='Input image file')
    args = ap.parse_args()

    if args.crop and not args.output:
        ap.error("--crop needs --output.")
        return 1

    if args.crop and args.debug:
        ap.error("--debug doesn't work with --crop.")
        return 1

    if args.crop and not (args.biggest or args.best):
        ap.error("--crop needs --biggest or --best.")
        return 1

    if args.resize and not args.crop:
        ap.error("--resize needs --crop.")
        return 1

    if args.grayscale and not args.crop:
        ap.error("--grayscale needs --crop.")
        return 1

    check_profiles()

    im = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
    if im is None:
        fatal("cannot load input image {}".format(args.file))

    im = cv2.equalizeHist(im)
    side = math.sqrt(im.size)
    minlen = int(side / 20)
    maxlen = int(side / 2)
    flags = cv2.cv.CV_HAAR_DO_CANNY_PRUNING

    # optimize queries when possible
    if args.biggest or args.query:
        flags |= cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT

    # frontal faces
    cc = cv2.CascadeClassifier(PROFILES['HAAR_FRONTALFACE_ALT2'])
    features = cc.detectMultiScale(im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
    if args.query:
        return 0 if len(features) else 2

    # compute scores
    scores = []
    best = None
    if len(features) and (args.debug or args.best or args.biggest):
        scores, best = rank(im, features)

    # debug features
    if args.output:
        im = cv2.imread(args.file)
        font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
        fontHeight = cv2.cv.GetTextSize("", font)[0][1] + 5

        for i in range(len(features)):
            if best is not None and i != best and not args.debug:
                continue

            rect = features[i]

            # scale width and height
            width = int(rect[2] * args.scale)
            height = int(rect[3] * args.scale)
            # calculate center of the face
            center_x = int(rect[0] + rect[2] / 2)
            center_y = int(rect[1] + rect[3] / 2) 
            # calculate new left top
            x = center_x - int(width / 2)
            y = center_y - int(height / 2)

            if args.center:
                print("{} {}".format(center_x, center_y))
            else:
                print("{} {} {} {}".format(x, y, width, height))

            if args.crop:
                # crop image
                im = im[y:(y + height), x:(x + width)]
                # resize if needed
                if args.resize:
                    im = cv2.resize(im, (int(width * args.resize / height), args.resize))
                # convert to grayscale if needed
                if args.grayscale:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
            else:
                fg = (0, 255, 255) if i == best else (255, 255, 255)
                cv2.rectangle(im, (x, y), (x + width, y + height), (0, 0, 0), 4)
                cv2.rectangle(im, (x, y), (x + width, y + height), fg, 2)

                if args.debug:
                    lines = []
                    for k, v in scores[i].iteritems():
                        lines.append("{}: {}".format(k, v))
                    h = y + height + fontHeight
                    for line in lines:
                        cv2.cv.PutText(cv2.cv.fromarray(im), line, (x, h), font, fg)
                        h += fontHeight

        if len(features):
            cv2.imwrite(args.output, im)

    return 0


if __name__ == '__main__':
    sys.exit(__main__())
