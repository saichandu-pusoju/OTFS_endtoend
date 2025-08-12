
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
import MIMO

EbdB = np.arange(-10, 10.1, 2.0)
Eb = 10 ** (EbdB / 10)
No = 1
SNR = 2 * Eb / No
SNRdB = 10 * np.log10(SNR)

M = 32
N = 16
eyeM = np.identity(M)
nTaps = 5
DelayTaps = np.arange(0, 5)
Ncp = np.max(DelayTaps)
DopplerTaps = np.array([0, 1, 2, 3, 4])
Ncp = np.max(DelayTaps)

BER_MMSE = np.zeros(len(Eb))
BER_ZF = np.zeros(len(Eb))

ITER = 100
F_M = 1 / np.sqrt(M) * MIMO.DFTmat(M)
F_N = 1 / np.sqrt(N) * MIMO.DFTmat(N)

for ite in range(ITER):
    print(ite)
    XddBits = nr.randint(2, size=(M, N))
    h = (nr.normal(0.0, 1.0, nTaps) + 1j * nr.normal(0.0, 1.0, nTaps)) / np.sqrt(2)
    Hmat = np.zeros((M * N, M * N))
    omega = np.exp(1j * 2 * np.pi / M / N)

    for tx in range(nTaps):
        DiagMat = np.diag(omega ** (np.arange(0, M * N) * DopplerTaps[tx]))
        CircMat = np.roll(np.identity(M * N), -DelayTaps[tx], axis=1)
        Hmat = Hmat + h[tx] * np.matmul(CircMat, DiagMat)

    Heff = nl.multi_dot([np.kron(F_N, eyeM), Hmat, np.kron(MIMO.H(F_N), eyeM)])
    ChNoise = nr.normal(0.0, np.sqrt(No / 2), M * N) + 1j * nr.normal(0.0, np.sqrt(No / 2), M * N)

    for ix in range(Eb.size):
        X_DD = np.sqrt(Eb[ix]) * (2 * XddBits - 1)

        # ZAK Transform
        S_mat = nl.multi_dot([eyeM, X_DD, MIMO.H(F_N)])
        TxSamples = S_mat.flatten('F')
        TxSamCP = np.concatenate((TxSamples[M * N - Ncp : M * N], TxSamples))
        RxSamCP = 0

        for tx in range(nTaps):
            Doppler = np.exp(1j * 2 * np.pi / M * np.arange(-Ncp, M * N) * DopplerTaps[tx] / N)
            RxSamCP = RxSamCP + h[tx] * np.roll(TxSamCP * Doppler, DelayTaps[tx])

        RxSamples = RxSamCP[Ncp : Ncp + M * N] + ChNoise
        R_mat = np.reshape(RxSamples, (M, N), order='F')

        # Inverse ZAK Transform
        Y_DD = nl.multi_dot([eyeM, R_mat, F_N])
        Y_DD = np.reshape(Y_DD, (M * N, 1), order='F')

        MMSErx = np.matmul(nl.inv(np.matmul(MIMO.H(Heff), Heff) + np.identity(M * N) / Eb[ix]), MIMO.H(Heff))
        xhatMMSE = np.matmul(MMSErx, Y_DD)
        DecodedBits = (np.real(xhatMMSE.flatten('F')) >= 0)
        BER_MMSE[ix] = BER_MMSE[ix] + np.sum(DecodedBits != XddBits.flatten('F'))

        xhatZF = np.matmul(nl.pinv(Heff), Y_DD)
        DecodedBits = (np.real(xhatZF.flatten('F')) >= 0)
        BER_ZF[ix] = BER_ZF[ix] + np.sum(DecodedBits != XddBits.flatten('F'))

BER_MMSE = BER_MMSE / M / N / ITER
BER_ZF = BER_ZF / M / N / ITER

plt.yscale('log')
plt.plot(EbdB, BER_MMSE, 'g-s')
plt.plot(EbdB, BER_ZF, 'r-o')
plt.grid(1, which='both')

plt.suptitle('OTFS BER v/s SNR')
plt.legend(["MMSE", "ZF"], loc="lower left")
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.show()
