import json
import argparse
import os
from os.path import join, exists

import SimpleITK as sitk


def dilatatelabel(image, backgroundvalue, foregroundvalue):

    filter = sitk.BinaryDilateImageFilter()
    filter.SetBackgroundValue(backgroundvalue)
    filter.SetForegroundValue(foregroundvalue)
    filter.SetKernelType(sitk.sitkBSpline)
    filter.SetKernelRadius([1, 1, 1])
    dilatatedlabel = filter.Execute(image)

    return dilatatedlabel


def sumimage(image1, image2):

    filter = sitk.AddImageFilter()
    sumimage = filter.Execute(image1, image2)

    return sumimage


def _nrrd2nifti(idir, imgid, frameid, id, odir, annulus=False):
    """
        Convert philips sequences, exported from 3DSlicer. from nrrd format
        to nifti and extract for them the annotated frame (3D volume).
        Convert the corresponding GT annotation (label1: annulus, label2:
        anterior leaflet, label3: posterior leaflet) from nrrd format to nifti.
        Enable or disable the annulus as label mask (for NTNU collaboration).
    """

    imgsitk = sitk.ReadImage(join(idir, f'images/{imgid:03d}.nrrd'))
    gtsitk = sitk.ReadImage(join(idir, f'masks_es/{imgid:03d}_{frameid:03d}.nrrd'))

    imgnp = sitk.GetArrayFromImage(imgsitk)[:, :, :, frameid]

    imgsitk_ = sitk.GetImageFromArray(imgnp)
    imgsitk_.CopyInformation(imgsitk)
    imgsitk_ = sitk.Cast(imgsitk_, sitk.sitkFloat32)  # recast before saving

    if not exists(join(args.odir, 'images')):
        os.makedirs(join(args.odir, 'images'))

    sitk.WriteImage(imgsitk_, join(odir, f'images/{id:03d}.nii.gz'))

    if annulus:

        annulus = sitk.BinaryThreshold(gtsitk, lowerThreshold=1,
                                       upperThreshold=1, insideValue=3, outsideValue=0)

        anteriorleaflet = sitk.BinaryThreshold(gtsitk, lowerThreshold=2,
                                               upperThreshold=2, insideValue=1, outsideValue=0)

        posteriorleaflet = sitk.BinaryThreshold(gtsitk, lowerThreshold=3,
                                                upperThreshold=3, insideValue=2, outsideValue=0)
        thickannulus = dilatatelabel(annulus, 0.0, 3.0)
        leaflets = sumimage(anteriorleaflet, posteriorleaflet)
        valve = sumimage(thickannulus, leaflets)

        anteriorleaflet = sitk.BinaryThreshold(valve, lowerThreshold=1,
                                               upperThreshold=1, insideValue=1, outsideValue=0)

        posteriorleaflet = sitk.BinaryThreshold(valve, lowerThreshold=2,
                                                upperThreshold=2, insideValue=2, outsideValue=0)

        annulus = sitk.BinaryThreshold(valve, lowerThreshold=3, upperThreshold=5, insideValue=3, outsideValue=0)

        leaflets = sumimage(anteriorleaflet, posteriorleaflet)
        valve = sumimage(annulus, leaflets)

        anteriorleaflet = sitk.BinaryThreshold(valve, lowerThreshold=1,
                                               upperThreshold=1, insideValue=2, outsideValue=0)

        posteriorleaflet = sitk.BinaryThreshold(valve, lowerThreshold=2,
                                                upperThreshold=2, insideValue=3, outsideValue=0)
        annulus = sitk.BinaryThreshold(valve, lowerThreshold=3,
                                       upperThreshold=5, insideValue=1, outsideValue=0)

        leaflets = sumimage(anteriorleaflet, posteriorleaflet)
        gtsitk_ = sumimage(annulus, leaflets)
        gtsitk_ = sitk.Cast(gtsitk_, sitk.sitkUInt8)

    else:

        gtnp = sitk.GetArrayFromImage(gtsitk)
        gtnp[gtnp == 1] = 0
        gtnp[gtnp == 2] = 1
        gtnp[gtnp == 3] = 2
        gtsitk_ = sitk.GetImageFromArray(gtnp)
        gtsitk_.CopyInformation(gtsitk)
        gtsitk_ = sitk.Cast(gtsitk_, sitk.sitkUInt8)

    if not exists(join(args.odir, 'masks')):
        os.makedirs(join(args.odir, 'masks'))

    sitk.WriteImage(gtsitk_, join(odir, f'masks/{id:03d}.nii.gz'))


def nrrd2nifti(idir, odir, jsondir, annulus):

    with open(join(jsondir, 'dataset.json'), 'r') as outfile:
        dataset = json.load(outfile)

    imgids, frameids, ids = ([dataset[key]['imageid'] for key in dataset.keys()],
                            [dataset[key]['frameid'] for key in dataset.keys()],
                            [dataset[key]['id'] for key in dataset.keys()])

    flat_imgid = [item for sublist in imgids for item in sublist]
    flat_frameid = [item for sublist in frameids for item in sublist]
    flat_id = [item for sublist in ids for item in sublist]

    for imgid, frameid, id in zip(flat_imgid, flat_frameid, flat_id):

        _nrrd2nifti(idir, imgid, frameid, id, odir, annulus=annulus)


if __name__ == '__main__':

    def str2bool(v):
        """
            Workaround to pass boolean to argparse
        """
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Philips dataset preprocessing')
    parser.add_argument('idir', type=str, default='./', help='input directory (where nrrd files are)')
    parser.add_argument('odir', type=str, default='./input', help='output directory (input folder in project dir)')
    parser.add_argument('jsondir', type=str, default='./', help='input directory for the json files')
    parser.add_argument('--annulus', type=str2bool, default=False, help='consider the annulus as label in the GT')
    args = parser.parse_args()

    nrrd2nifti(args.idir, args.odir, args.jsondir, args.annulus)
