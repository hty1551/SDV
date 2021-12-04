# （用户x商品）    # 为0表示该用户未评价此商品，即可以作为推荐商品
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
 
# 替代上面的standEst(功能) 该函数用SVD降维后的矩阵来计算评分
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])  #将奇异值向量转换为奇异值矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # 降维方法 通过U矩阵将物品转换到低维空间中 （商品数行x选用奇异值列）
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        # 这里需要说明：由于降维后的矩阵与原矩阵代表数据不同（行由用户变为了商品），所以在比较两件商品时应当取【该行所有列】 再转置为列向量传参
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        # print('%d 和 %d 的相似度是: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
 
# 结果测试如下：
myMat = mat(loadExData2())
result1 = recommend(myMat,1,estMethod=svdEst)   # 需要传参改变默认函数
print(result1)
result2 = recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
print(result2)