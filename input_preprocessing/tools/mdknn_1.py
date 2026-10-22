import os, argparse, sys, logging, logging.handlers
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import poisson, itemfreq
from scipy.stats import ks_2samp
import scipy.special as special
from scipy.interpolate import interp1d, splrep, splev
from sklearn import cluster
from sklearn.neighbors import KDTree



log = logging.getLogger(__name__)


def load_TAD(path):














    import pandas as pd
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.iloc[:, :3]
    df.columns = ['chr', 'start', 'end']
    df['chr'] = df['chr'].astype('str')
    if df.shape[0] > 0 and not df.iloc[0, 0].startswith('chr'):
        df['chr'] = 'chr' + df['chr']
    return df


def manipulation(matrix, start = 0):









































    mask = matrix.sum(axis = 0) == 0
    index = list(np.where(mask)[0])

    temp = np.delete(matrix, index, 0)

    newM = np.delete(temp, index, 1)
    mapidx = dict(zip(np.arange(len(newM)),
                      np.where(np.logical_not(mask))[0] + start))
    convert = [mapidx, matrix.shape[0]]

    return newM, convert


def _fitting(x, y):


    ip = interp1d(x, y)

    times = np.arange(2, 4)
    scheck = x.size / times
    snum = scheck[scheck > 6][-1] if (scheck > 6).sum() > 0 else x.size
    snum = int(snum)
    xi = np.linspace(x.min(), x.max(), snum)
    yi = ip(xi)


    tcl = splrep(xi, yi)
    ys = splev(x, tcl)


    dy1 = np.gradient(ys)


    m = (dy1[1:] >= 0) & (dy1[:-1] <= 0)
    if len(np.where(m)[0]) != 0:
        i = np.where(m)[0][0]
        ys[x > x[i]] = ys[i]

    return ys


class Core(object):






















































    def __init__(self, matrix, k=1, left = 0):
        if k != 1:
            raise ValueError("MD-KNN uses a fixed k=1; other values are not supported")
        matrix[np.isnan(matrix)] = 0
        self.k = 1

        self.matrix = matrix


        nonzero = matrix[matrix.nonzero()]
        if np.median(nonzero) < 1:
            min_nonzero = nonzero.min()
            scale = 1 / min_nonzero
            matrix = matrix * scale


        self.newM, self.convert = manipulation(matrix, left)
        self._convert = np.array(list(self.convert[0].values()))


        Len = self.newM.shape[0]
        idiag = np.arange(0, Len)
        iIFs = []
        for i in idiag:
            temp = np.diagonal(self.newM, offset = i)
            iIFs.append(temp.mean())
        iIFs = np.array(iIFs)

        idx = np.where(iIFs > 0)[0][0]

        self._start = idx
        IFs = iIFs[idx:]
        diag = idiag[idx:]

        self._Ed = _fitting(diag, IFs)

    def longrange(self, pw = 2, ww = 5, top = 0.7, ratio = 0.05):
































        dim = self.newM.shape[0]

        ps = 2 * pw + 1
        ws = 2 * ww + 1
        bs = 2 * pw + 1

        start = ww if (ww > self._start) else self._start

        upper = np.triu(self.newM, k = start)
        bUpper = np.triu(self.newM, k = 0)


        expM = np.zeros((dim + ww*2, dim + ww*2))
        expBM = np.zeros((dim + ww*2, dim + ww*2))
        expM[ww:-ww, ww:-ww] = upper
        expBM[ww:-ww, ww:-ww] = bUpper

        tm = np.all((expBM == 0), axis = 0)
        Mask = np.zeros((dim + ww*2, dim + ww*2), dtype = bool)
        Mask[:,tm] = True
        Mask[tm,:] = True
        expCM = np.ones_like(expM, dtype = int)
        expCM[Mask] = 0


        EM_idx = np.triu_indices(dim, k = start)
        EM_value = self._Ed[EM_idx[1] - EM_idx[0] - self._start]
        EM = np.zeros((dim, dim))
        EM[EM_idx] = EM_value

        expEM = np.zeros((dim + ww*2, dim + ww*2))
        expEM[ww:-ww, ww:-ww] = EM



        OPool_w = {}
        EPool_w = {}
        ss = range(ws)
        for i in ss:
            for j in ss:
                OPool_w[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_w[(i,j)] = expEM[i:(dim+i), j:(dim+j)]

        OPool_p = {}
        EPool_p = {}
        ss = range(ww-pw, ps+ww-pw)
        for i in ss:
            for j in ss:
                OPool_p[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_p[(i,j)] = expEM[i:(dim+i), j:(dim+j)]


        OPool_b = {}
        OPool_bc = {}
        ss = range(ww-pw, bs+ww-pw)
        for i in ss:
            for j in ss:
                OPool_b[(i,j)] = expBM[i:(dim+i), j:(dim+j)]
                OPool_bc[(i,j)] = expCM[i:(dim+i), j:(dim+j)]


        bS = np.zeros((dim, dim))
        bE = np.zeros((dim, dim))
        for w in OPool_w:
            if (w[0] != ww) and (w[1] != ww):
                bS += OPool_w[w]
                bE += EPool_w[w]
        for p in OPool_p:
            if (p[0] != ww) and (p[1] != ww):
                bS -= OPool_p[p]
                bE -= EPool_p[p]

        bE[bE==0] = 1
        bR = bS / bE


        cEM = EM * bR
        self.cEM = cEM


        BM = np.zeros((dim, dim))
        CM = np.zeros((dim, dim), dtype = int)

        for b in OPool_b:
            BM += OPool_b[b]
            CM += OPool_bc[b]

        mBM = np.zeros_like(BM)
        Mask = CM != 0
        mBM[Mask] = BM[Mask] / CM[Mask]


        self.fE = np.zeros_like(self.cEM)
        mask = self.cEM != 0
        self.fE[mask] = upper[mask] / self.cEM[mask]


        Poisses = poisson(cEM)
        Ps = 1 - Poisses.cdf(upper)
        self.Ps = Ps
        rmBM = mBM[EM_idx]

        top_idx = np.argsort(rmBM)[np.int(np.floor((1-top)*rmBM.size)):]

        rPs = Ps[EM_idx][top_idx]
        Rnan = np.logical_not(np.isnan(rPs))
        RrPs = rPs[Rnan]
        Np = np.int(np.ceil(ratio/top*RrPs.size))
        sig_idx = np.argsort(RrPs)[:Np]
        if sig_idx.size > 0:
            self.pos = np.r_['1,2,0', EM_idx[0][top_idx][Rnan][sig_idx], EM_idx[1][top_idx][Rnan][sig_idx]]
            self._pos = self.pos
            self.pos = self.convertPos(self.pos)
        else:
            self.pos = np.array([])

        self.Np = len(self.pos)
        self._area = EM_idx[0].size

    def pos2range(self, tad_chrom, tad_start, tad_end, binsize):



        import pandas as pd
        df = pd.DataFrame(self.pos, columns=['bin1_id', 'bin2_id'])
        def to_range(bin_ids):
            start = tad_start + binsize * bin_ids
            end = start + binsize
            range_ = tad_chrom + ":" + start.astype('str') + "-" + end.astype('str')
            return range_
        df['tad_range'] = "{}:{}-{}".format(tad_chrom, tad_start, tad_end)
        df['bin1_range'] = to_range(df['bin1_id'])
        df['bin2_range'] = to_range(df['bin2_id'])
        return df

    def convertMatrix(self, M):





        idx = sorted(self.convert[0].values())
        newM = np.zeros((self.convert[1], self.convert[1]), dtype=M.dtype)
        y,x = np.meshgrid(idx, idx)
        newM[x,y] = M

        return newM

    def convertPos(self, pos):




        new_x = self._convert[pos[:, 0]]
        new_y = self._convert[pos[:, 1]]
        new_pos = np.c_[new_x, new_y]
        return new_pos

    def MDKNN(self):







        k = 1
        if self.Np < k + 5:
            self.mean_dist = np.nan
            self.mean_dist_all = np.nan
            self.AP = np.nan
            self.local_ap = np.nan
            return

        self._kdtree = KDTree(self.pos)
        dist, ind = self._kdtree.query(self.pos, k=k+1)
        self._DKNN = dist[:, 1:]
        self._KNN = ind[:, 1:]

        self.mean_dist_all = self._DKNN.mean(axis=1)
        self.mean_dist = self.mean_dist_all.mean()

        self.DoD = self.mean_dist


class Compare(object):












    def __init__(self, core1, core2):
        self.core1 = core1
        self.core2 = core2

    def compare(self):







        dist1 = self.core1.mean_dist_all
        dist2 = self.core2.mean_dist_all
        D, pvalue = ks_2samp(dist1, dist2)
        diff = dist2.mean() - dist1.mean()
        self.D = D
        self.pvalue = pvalue
        self.diff = diff


def getmatrix(inter, l_bin, r_bin):






















    inter_matrix = np.zeros((r_bin - l_bin, r_bin - l_bin), dtype = float)

    mask = (inter['bin1'] >= l_bin) & (inter['bin1'] < r_bin) & \
           (inter['bin2'] >= l_bin) & (inter['bin2'] < r_bin)
    inter_extract = inter[mask]


    for i in inter_extract:

        if i['bin1'] != i['bin2']:
            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']
            inter_matrix[i['bin2'] - l_bin][i['bin1'] - l_bin] += i['IF']
        else:

            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']

    return inter_matrix

def getargs():

    parser = argparse.ArgumentParser(usage = '%(prog)s [options]\n\n'
                                     'MDKNN -- Mean Distance of the Nearest Neighbor (fixed k=1)',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('-O', '--output',
                        help = 'Output file name.')


    group_1 = parser.add_argument_group(title = 'Relate to the input')
    group_1.add_argument('-p', '--path', default = '.',
                         help = 'Path to the cool URI')
    group_1.add_argument('-p2', '--path-2',
                         help = 'Path to the cool URI')
    group_1.add_argument('-t', '--tad-file', help = 'Path to the TAD file.')
    group_1.add_argument('--out-long-range', help = 'Output the position of long-range points.')


    group_3 = parser.add_argument_group(title='Feature calculation')
    group_3.add_argument('-k', type=int, default=1, choices=[1], help='Number of nearest neighbors (fixed at 1).')
    group_3.add_argument('--pw', type = int, default = 2, help = '''Width of the interaction
                         region surrounding the peak. According to experience, we set it
                         to 1 at 20 kb, 2 at 10 kb, and 4 at 5 kb.''')
    group_3.add_argument('--ww', type = int, default = 5, help = '''Width of the donut region
                         Set it to 3 at 20 kb, 5 at 10 kb, and 7 at 5 kb.''')
    group_3.add_argument('--top', type = float, default = 0.7, help = 'Parameter for noisy '
                         'interaction filtering. By default, 30 percent noisy interactions'
                         ' will be eliminated.')
    group_3.add_argument('--ratio', type = float, default = 0.05, help = 'Specifies the sample'
                         ' ratio of significant interactions for TAD.')
    group_3.add_argument('--gap', type = float, default = 0.2, help = 'Maximum gap ratio.')


    commands = sys.argv[1:]
    if not commands:
        commands.append('-h')
    args = parser.parse_args(commands)

    return args, commands


def get_gap_ratio(matrix):
    matrix[np.isnan(matrix)] = 0
    newM, _ = manipulation(matrix)
    if len(matrix) > 0:

        gaps = 1 - len(newM) / len(matrix)
    else:
        gaps = 1.0
    return newM, gaps



def pipe(args, logger):



    arglist = ['# ARGUMENT LIST:',
               '# output file name = {0}'.format(args.output),
               '# Hi-C path = {0}'.format(args.path),
               '# TAD source file = {0}'.format(args.tad_file),
               '# Peak window width = {0}'.format(args.pw),
               '# Donut width = {0}'.format(args.ww),
               '# Noise filtering ratio = {0}'.format((1 - args.top)),
               '# Significant interaction ratio = {0}'.format(args.ratio),
               '# Maximum gap ratio = {0}'.format(args.gap)]

    argtxt = '\n'.join(arglist)
    logger.info('\n' + argtxt)

    logger.info('Read Hi-C data ...')
    cool = cooler.Cooler(args.path)


    logger.info('Read external TAD data ...')

    TADs = load_TAD(args.tad_file)


    header = ['ChromID', 'Start', 'End', 'DoD', 'Gap-Ratio']
    out_rows = []

    if args.out_long_range:

        long_range_pos = []

    logger.info('Calculate feature for each TAD ...')
    for i, row in tqdm(TADs.iterrows(), total=TADs.shape[0]):
        chr_, start, end = row.chr, row.start, row.end


        balance = 'weight' in cool.bins().keys()
        try:
            matrix = cool.matrix(balance=balance, sparse=False)\
                       .fetch((chr_, start, min(end, cool.chromsizes[chr_])))
        except KeyError:
            if chr_.startswith('chr'):
                chr_ = chr_[3:]
            else:
                chr_ = 'chr'+chr_
            matrix = cool.matrix(balance=balance, sparse=False)\
                       .fetch((chr_, start, min(end, cool.chromsizes[chr_])))

        newM, gaps = get_gap_ratio(matrix)
        if (gaps < args.gap) and (newM.shape[0] > (args.ww * 2 + 1)):
            core = Core(matrix, k=args.k)

            core.longrange(pw=args.pw, ww=args.ww, top=args.top, ratio=args.ratio)

            core.MDKNN()


            if core.mean_dist_all is not np.nan:
                o_row = [chr_, start, end, core.DoD, gaps]
            else:
                o_row = [chr_, start, end, np.nan, gaps]

            if args.out_long_range:
                df = core.pos2range(chr_, start, end, cool.binsize)
                long_range_pos.append(df)
        else:

            o_row = [chr_, start, end, np.nan, gaps]
        out_rows.append(o_row)
    logger.info('Done!')
    out_df = pd.DataFrame(out_rows)
    out_df.columns = header

    if args.out_long_range:
        logger.info("Write long-range point positions to %s ...", args.out_long_range)
        long_range_pos_df = pd.concat(long_range_pos)
        long_range_pos_df.to_csv(args.out_long_range, sep="\t", index=False)

    logger.info('Write results to %s ...', args.output)
    out_df.to_csv(args.output, na_rep='NA', sep='\t', index=False)
    logger.info('Done!\n')


def pipe_compare(args, logger):

    arglist = ['# ARGUMENT LIST:',
               '# output file name = {0}'.format(args.output),
               '# Comparision mode: True',
               '# Hi-C path 1 = {0}'.format(args.path),
               '# Hi-C path 2 = {0}'.format(args.path_2),
               '# TAD source file = {0}'.format(args.tad_file),
               '# Peak window width = {0}'.format(args.pw),
               '# Donut width = {0}'.format(args.ww),
               '# Noise filtering ratio = {0}'.format((1 - args.top)),
               '# Significant interaction ratio = {0}'.format(args.ratio),
               '# Maximum gap ratio = {0}'.format(args.gap)]

    argtxt = '\n'.join(arglist)
    logger.info('\n' + argtxt)

    logger.info('Read Hi-C data ...')
    cool_1 = cooler.Cooler(args.path)
    cool_2 = cooler.Cooler(args.path_2)
    assert cool_1.binsize == cool_2.binsize, "cooler1 and cooler2's binsize must same."


    logger.info('Read external TAD data ...')
    TADs = load_TAD(args.tad_file)

    out_rows = []

    header = ['ChromID', 'Start', 'End', 'DoD1', 'DoD2', 'D', 'p-value', 'padj', 'Gap-Ratio-1', 'Gap-Ratio-2']

    if args.out_long_range:

        long_range_pos = []

    logger.info('Calculate feature for each TAD ...')
    for i, row in tqdm(TADs.iterrows(), total=TADs.shape[0]):
        chr_, start, end = row.chr, row.start, row.end


        balance = 'weight' in cool_1.bins().keys()
        try:
            matrix_1 = cool_1.matrix(balance=balance,  sparse=False)\
                       .fetch((chr_, start, min(end, cool_1.chromsizes[chr_])))
        except KeyError:
            if chr_.startswith('chr'):
                chr_ = chr_[3:]
            else:
                chr_ = 'chr'+chr_
            matrix_1 = cool_1.matrix(balance=balance,  sparse=False)\
                       .fetch((chr_, start, min(end, cool_1.chromsizes[chr_])))
        balance = 'weight' in cool_2.bins().keys()
        matrix_2 = cool_2.matrix(balance=balance,  sparse=False)\
                   .fetch((chr_, start, min(end, cool_2.chromsizes[chr_])))

        newM_1, gaps_1 = get_gap_ratio(matrix_1)
        newM_2, gaps_2 = get_gap_ratio(matrix_2)

        if (gaps_1 < args.gap) and (newM_1.shape[0] > (args.ww * 2 + 1)) and \
           (gaps_2 < args.gap) and (newM_2.shape[0] > (args.ww * 2 + 1)):

            core_1 = Core(matrix_1, k=args.k)
            core_1.longrange(pw = args.pw, ww = args.ww, top = args.top, ratio = args.ratio)
            core_1.MDKNN()

            core_2 = Core(matrix_2, k=args.k)
            core_2.longrange(pw = args.pw, ww = args.ww, top = args.top, ratio = args.ratio)
            core_2.MDKNN()
            comp = Compare(core_1, core_2)

            if args.out_long_range:
                df1 = core_1.pos2range(chr_, start, end, cool_1.binsize)
                df2 = core_2.pos2range(chr_, start, end, cool_2.binsize)
                tad_ = df1.pop('tad_range'); df2.pop('tad_range')
                df1.add_prefix('sample1_'); df2.add_prefix('sample2_')
                df = pd.concat([df1, df2, tad_], axis=1)
                long_range_pos.append(df)

            if (core_1.mean_dist_all is np.nan) or (core_2.mean_dist_all is np.nan):

                o_row = [chr_, start, end, np.nan, np.nan, np.nan, np.nan, np.nan, gaps_1, gaps_2]
            else:
                comp.compare()
                pvalue, D = comp.pvalue, comp.D
                o_row = [chr_, start, end, core_1.DoD, core_2.DoD, D, pvalue, np.nan, gaps_1, gaps_2]
        else:

            o_row = [chr_, start, end, np.nan, np.nan, np.nan, np.nan, np.nan, gaps_1, gaps_2]

        out_rows.append(o_row)

    out_df = pd.DataFrame(out_rows)
    out_df.columns = header

    logger.info("Perform multiple test correction with fdr_bh method.")
    pvals = out_df['p-value']
    pvals_ = pvals[~pvals.isna()]
    _, padjs, _, _ = multipletests(pvals_, method='fdr_bh')
    out_df.loc[pvals_.index, 'padj'] = padjs

    logger.info('Done!')
    logger.info('Write results to %s ...', args.output)
    out_df.to_csv(args.output, sep='\t', na_rep='NA', index=False)

    if args.out_long_range:
        logger.info("Write long-range point positions to %s ...", args.out_long_range)
        long_range_pos_df = pd.concat(long_range_pos)
        long_range_pos_df.to_csv(args.out_long_range, sep="\t", index=False)

    logger.info('Done!\n')


def main():

    args, commands = getargs()

    if commands[0] not in ['-h', '-v', '--help', '--version']:

        logger = logging.getLogger()
        logger.setLevel(10)
        console = logging.StreamHandler()
        filehandler = logging.handlers.RotatingFileHandler('mdknn.log',
                                                           maxBytes = 30000,
                                                           backupCount = 5)

        console.setLevel('INFO')
        filehandler.setLevel('DEBUG')

        formatter = logging.Formatter(fmt = '%(name)-14s %(levelname)-7s @ %(asctime)s: %(message)s',
                                      datefmt = '%m/%d/%y %H:%M:%S')

        console.setFormatter(formatter)
        filehandler.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(filehandler)


        comparision = False
        if args.path_2:
            comparision = True

        if comparision:
            pipe_compare(args, logger)
        else:
            pipe(args, logger)


if __name__ == '__main__':
    main()
