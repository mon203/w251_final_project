import os
from pathlib import Path
from tqdm import tqdm
import trimesh
import numpy as np


def get_all_files(directory, pattern):
    return [f for f in Path(directory).glob(pattern)]


def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)


def getGreedyPerm(X, M, Verbose = False):
    """
    Purpose: Naive O(NM) algorithm to do the greedy permutation
    :param X: Nxd array of Euclidean points
    :param M: Number of points in returned permutation
    :returns: (permutation (N-length array of indices), \
            lambdas (N-length array of insertion radii))
    """
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = getCSM(X[0, :][None, :], X).flatten()
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getCSM(X[idx, :][None, :], X).flatten())
        if Verbose:
            interval = int(0.05*M)
            if i%interval == 0:
                print("Greedy perm %i%s done..."%(int(100.0*i/float(M)), "%"))
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas}


def save_keypoint_files(file, keypoints):
    file_name = file.parent.joinpath(file.stem + "_keypoints.ply")
    data = """ply 
format ascii 1.0 
comment lines 
element vertex 9
property float x 
property float y 
property float z 
end_header\n"""
    for line in keypoints:
        data += (" ".join(str(x) for x in line))
        data += "\n"

    with open(file_name, "w") as f:
        f.write(data)


def process_keypoints(mesh_files):
    for file in tqdm(mesh_files):
        mesh = trimesh.load_mesh(file, process=False)

        center = mesh.bounding_box.center_mass
        vertices = np.array(mesh.vertices.tolist())
        X = np.concatenate((np.array([center]), vertices), axis=0)
        keypoints = getGreedyPerm(X, 9, Verbose=False)
        Y = keypoints['Y']
        save_keypoint_files(file, Y)


def main():
    mesh_dir = r"D:\bop_toolkit\data\lm\blenderproc\bop_data\lm\models"
    mesh_files = get_all_files(mesh_dir, "*.ply")

    process_keypoints(mesh_files)

    print("Done")


if __name__ == '__main__':
    main()
