import numpy as np

def findNumberIn2DArray(matrix, target) -> bool:
    n = len(matrix)
    m = len(matrix[0])
    result = 0
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == target:
                result = 1
    return result

s = findNumberIn2DArray([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 100)
a=1