using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.IO;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic;

namespace TestInfer
{
    class Program
    {
        // This function calculates the z node responsibilities using exponential family form
        // function takes observed r and all its estimated Gammas as inputs
        // function returns array of z = 1 probabilities for each r observation
        static double[] ZResponsibilities(double[] Gammas, double[] r)
        {
            double mean0 = -1.0;
            double mean1 = 1.0;

            double[] z = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                double g0 = Gammas[i];
                double x = r[i];

                double z0 = (x * g0 * mean0) - (Math.Pow(x, 2) * g0 / 2.0) + (0.5 * (Math.Log(g0) - g0 * Math.Pow(mean0, 2) - Math.Log(2.0 * Math.PI)));
                double z1 = (x * g0 * mean1) - (Math.Pow(x, 2) * g0 / 2.0) + (0.5 * (Math.Log(g0) - g0 * Math.Pow(mean1, 2) - Math.Log(2.0 * Math.PI)));

                double b = Math.Max(z0, z1);
                z0 = Math.Exp(z0 - b);
                z1 = Math.Exp(z1 - b);

                z1 = (z1 / (z0 + z1));

                z[i] = z1;
            }

            return z;
        }
        // This function is the Noise precision estimation model (using VMP)
        // function takes observed r and responsibilities z as inputs
        // function returns 2D array of Gamma posterior parameters for each r observation
        static Tuple<double[,], double[], int> NoiseEstimation(double Nmax, double aPrior, double bPrior, double[] prevGammaNatural, double[] r, double[] z, int bitCount)
        {
            double mean0 = -1.0;
            double mean1 = 1.0;

            double[,] ab = new double[r.Length, 2];

            //double Nmax = 20;

            double[] GammaNaturalPrior = { -bPrior, aPrior - 1.0 };
            double[] GammaNatural = { prevGammaNatural[0], prevGammaNatural[1] };
            double[] GammaNaturalPosterior = { 0.0, 0.0 };

            for (int i = 0; i < r.Length; i++)
            {
                double zTrue = z[i];
                double zFalse = 1.0 - zTrue;
                //maybe this should be OR instead and not AND
                //or change this to if ((i + bitcount) > Nmax)
                if (i + bitCount > Nmax)
                {
                    GammaNatural[0] = (GammaNatural[0] + ((zTrue * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean1) + Math.Pow(mean1, 2))) +
                                                          (zFalse * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean0) + Math.Pow(mean0, 2))))) * (Nmax / (Nmax + 1));
                    GammaNatural[1] = (GammaNatural[1] + (zTrue * 0.5) +
                                                         (zFalse * 0.5)) * (Nmax / (Nmax + 1));
                }
                else
                {
                    GammaNatural[0] += ((zTrue * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean1) + Math.Pow(mean1, 2))) +
                                        (zFalse * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean0) + Math.Pow(mean0, 2))));
                    GammaNatural[1] += ((zTrue * 0.5) + (zFalse * 0.5));
                }

                GammaNaturalPosterior[0] = GammaNaturalPrior[0] + GammaNatural[0];
                GammaNaturalPosterior[1] = GammaNaturalPrior[1] + GammaNatural[1];

                ab[i, 0] = GammaNaturalPosterior[1] + 1.0;      // a parameter
                ab[i, 1] = -1.0 * GammaNaturalPosterior[0];     // b parameter

                bitCount++;
            }

            return new Tuple<double[,], double[], int>(ab, GammaNatural, bitCount);
        }
        // Here we specify the model by using a baseclass
        public class CyclistBase
        {
            public InferenceEngine InferenceEngine;

            protected VariableArray<bool> v;
            protected VariableArray<Bernoulli> vPriors;

            public virtual void CreateModel()
            {
                Range n = new Range(540);

                v = Variable.Array<bool>(n).Named("v");

                vPriors = Variable.Array<Bernoulli>(n);

                v[n] = Variable.Random<bool, Bernoulli>(vPriors[n]);

                //88-44-0.5
                //Variable.ConstrainEqual(v[0] != v[3] != v[4] != v[7] != v[10] != v[12] != v[19] != v[21] != v[22] != v[25] != v[27] != v[31] != v[33] != v[36] != v[39] != v[41] != v[43] != v[45] != v[46], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[5] != v[6] != v[11] != v[13] != v[18] != v[20] != v[23] != v[24] != v[26] != v[30] != v[32] != v[37] != v[38] != v[40] != v[42] != v[44] != v[47], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[5] != v[7] != v[8] != v[11] != v[14] != v[16] != v[19] != v[22] != v[24] != v[29] != v[30] != v[32] != v[35] != v[39] != v[42] != v[44] != v[46] != v[48], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[4] != v[6] != v[9] != v[10] != v[15] != v[17] != v[18] != v[23] != v[25] != v[28] != v[31] != v[33] != v[34] != v[38] != v[43] != v[45] != v[47] != v[49], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[5] != v[9] != v[11] != v[13] != v[15] != v[17] != v[19] != v[21] != v[26] != v[29] != v[31] != v[35] != v[37] != v[39] != v[41] != v[48] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[4] != v[8] != v[10] != v[12] != v[14] != v[16] != v[18] != v[20] != v[27] != v[28] != v[30] != v[34] != v[36] != v[38] != v[40] != v[49] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[8] != v[12] != v[15] != v[17] != v[20] != v[22] != v[24] != v[27] != v[29] != v[32] != v[34] != v[36] != v[40] != v[42] != v[45] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[9] != v[13] != v[14] != v[16] != v[21] != v[23] != v[25] != v[26] != v[28] != v[33] != v[35] != v[37] != v[41] != v[43] != v[44] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[52], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[53], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[6] != v[25] != v[32] != v[43] != v[45] != v[54], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[7] != v[24] != v[33] != v[42] != v[44] != v[55], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[12] != v[20] != v[23] != v[26] != v[35] != v[37] != v[41] != v[56], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[13] != v[21] != v[22] != v[27] != v[34] != v[36] != v[40] != v[57], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[9] != v[15] != v[17] != v[28] != v[58], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[8] != v[14] != v[16] != v[29] != v[59], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[25] != v[32] != v[38] != v[43] != v[45] != v[48] != v[60], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[24] != v[33] != v[39] != v[42] != v[44] != v[49] != v[61], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[21] != v[23] != v[26] != v[34] != v[37] != v[40] != v[62], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[20] != v[22] != v[27] != v[35] != v[36] != v[41] != v[63], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[5] != v[9] != v[14] != v[17] != v[28] != v[64], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[4] != v[8] != v[15] != v[16] != v[29] != v[65], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[25] != v[32] != v[42] != v[44] != v[46] != v[66], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[24] != v[33] != v[43] != v[45] != v[47] != v[67], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[22] != v[26] != v[37] != v[68], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[23] != v[27] != v[36] != v[69], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[6] != v[15] != v[41] != v[46] != v[70], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[7] != v[14] != v[40] != v[47] != v[71], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[25] != v[30] != v[33] != v[34] != v[43] != v[72], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[24] != v[31] != v[32] != v[35] != v[42] != v[73], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[27] != v[37] != v[51] != v[74], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[26] != v[36] != v[50] != v[75], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[7] != v[23] != v[41] != v[44] != v[76], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[6] != v[22] != v[40] != v[45] != v[77], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[28] != v[33] != v[35] != v[42] != v[78], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[29] != v[32] != v[34] != v[43] != v[79], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[25] != v[26] != v[37] != v[39] != v[80], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[24] != v[27] != v[36] != v[38] != v[81], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[14] != v[17] != v[20] != v[82], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[15] != v[16] != v[21] != v[83], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[6] != v[19] != v[23] != v[44] != v[84], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[7] != v[18] != v[22] != v[45] != v[85], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[11] != v[33] != v[41] != v[43] != v[86], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[10] != v[32] != v[40] != v[42] != v[87], Variable.Bernoulli(0));

                //124-80-0.35
                //Variable.ConstrainEqual(v[0] != v[3] != v[4] != v[7] != v[10] != v[12] != v[19] != v[21] != v[22] != v[25] != v[27] != v[31] != v[33] != v[36] != v[39] != v[41] != v[43] != v[45] != v[46], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[5] != v[6] != v[11] != v[13] != v[18] != v[20] != v[23] != v[24] != v[26] != v[30] != v[32] != v[37] != v[38] != v[40] != v[42] != v[44] != v[47], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[5] != v[7] != v[8] != v[11] != v[14] != v[16] != v[19] != v[22] != v[24] != v[29] != v[30] != v[32] != v[35] != v[39] != v[42] != v[44] != v[46] != v[48], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[4] != v[6] != v[9] != v[10] != v[15] != v[17] != v[18] != v[23] != v[25] != v[28] != v[31] != v[33] != v[34] != v[38] != v[43] != v[45] != v[47] != v[49], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[5] != v[9] != v[11] != v[13] != v[15] != v[17] != v[19] != v[21] != v[26] != v[29] != v[31] != v[35] != v[37] != v[39] != v[41] != v[48] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[4] != v[8] != v[10] != v[12] != v[14] != v[16] != v[18] != v[20] != v[27] != v[28] != v[30] != v[34] != v[36] != v[38] != v[40] != v[49] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[8] != v[12] != v[15] != v[17] != v[20] != v[22] != v[24] != v[27] != v[29] != v[32] != v[34] != v[36] != v[40] != v[42] != v[45] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[9] != v[13] != v[14] != v[16] != v[21] != v[23] != v[25] != v[26] != v[28] != v[33] != v[35] != v[37] != v[41] != v[43] != v[44] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[52], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[53], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[6] != v[25] != v[32] != v[43] != v[45] != v[54], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[7] != v[24] != v[33] != v[42] != v[44] != v[55], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[12] != v[20] != v[23] != v[26] != v[35] != v[37] != v[41] != v[56], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[13] != v[21] != v[22] != v[27] != v[34] != v[36] != v[40] != v[57], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[9] != v[15] != v[17] != v[28] != v[58], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[8] != v[14] != v[16] != v[29] != v[59], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[25] != v[32] != v[38] != v[43] != v[45] != v[48] != v[60], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[24] != v[33] != v[39] != v[42] != v[44] != v[49] != v[61], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[21] != v[23] != v[26] != v[34] != v[37] != v[40] != v[62], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[20] != v[22] != v[27] != v[35] != v[36] != v[41] != v[63], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[5] != v[9] != v[14] != v[17] != v[28] != v[64], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[4] != v[8] != v[15] != v[16] != v[29] != v[65], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[25] != v[32] != v[42] != v[44] != v[46] != v[66], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[24] != v[33] != v[43] != v[45] != v[47] != v[67], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[22] != v[26] != v[37] != v[68], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[23] != v[27] != v[36] != v[69], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[6] != v[15] != v[41] != v[46] != v[70], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[7] != v[14] != v[40] != v[47] != v[71], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[25] != v[30] != v[33] != v[34] != v[43] != v[72], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[24] != v[31] != v[32] != v[35] != v[42] != v[73], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[27] != v[37] != v[51] != v[74], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[26] != v[36] != v[50] != v[75], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[7] != v[23] != v[41] != v[44] != v[76], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[6] != v[22] != v[40] != v[45] != v[77], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[28] != v[33] != v[35] != v[42] != v[78], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[29] != v[32] != v[34] != v[43] != v[79], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[25] != v[26] != v[37] != v[39] != v[80], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[24] != v[27] != v[36] != v[38] != v[81], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[14] != v[17] != v[20] != v[82], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[15] != v[16] != v[21] != v[83], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[6] != v[19] != v[23] != v[44] != v[84], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[7] != v[18] != v[22] != v[45] != v[85], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[11] != v[33] != v[41] != v[43] != v[86], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[10] != v[32] != v[40] != v[42] != v[87], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[24] != v[26] != v[34] != v[88], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[25] != v[27] != v[35] != v[89], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[5] != v[20] != v[36] != v[90], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[4] != v[21] != v[37] != v[91], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[6] != v[8] != v[22] != v[44] != v[92], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[7] != v[9] != v[23] != v[45] != v[93], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[12] != v[14] != v[28] != v[94], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[13] != v[15] != v[29] != v[95], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[5] != v[9] != v[31] != v[96], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[4] != v[8] != v[30] != v[97], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[12] != v[16] != v[98], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[13] != v[17] != v[99], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[9] != v[39] != v[43] != v[100], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[8] != v[38] != v[42] != v[101], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[28] != v[36] != v[50] != v[102], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[29] != v[37] != v[51] != v[103], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[21] != v[26] != v[49] != v[104], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[20] != v[27] != v[48] != v[105], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[15] != v[44] != v[51] != v[106], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[14] != v[45] != v[50] != v[107], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[24] != v[29] != v[49] != v[108], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[25] != v[28] != v[48] != v[109], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[5] != v[23] != v[43] != v[110], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[4] != v[22] != v[42] != v[111], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[15] != v[30] != v[35] != v[112], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[14] != v[31] != v[34] != v[113], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[13] != v[25] != v[45] != v[114], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[12] != v[24] != v[44] != v[115], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[29] != v[31] != v[36] != v[116], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[28] != v[30] != v[37] != v[117], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[26] != v[46] != v[118], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[27] != v[47] != v[119], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[19] != v[21] != v[25] != v[120], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[18] != v[20] != v[24] != v[121], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[7] != v[15] != v[38] != v[122], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[6] != v[14] != v[39] != v[123], Variable.Bernoulli(0));

                //54-10-0.8
                //Variable.ConstrainEqual(v[0] != v[3] != v[4] != v[7] != v[10] != v[12] != v[19] != v[21] != v[22] != v[25] != v[27] != v[31] != v[33] != v[36] != v[39] != v[41] != v[43] != v[45] != v[46], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[5] != v[6] != v[11] != v[13] != v[18] != v[20] != v[23] != v[24] != v[26] != v[30] != v[32] != v[37] != v[38] != v[40] != v[42] != v[44] != v[47], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[5] != v[7] != v[8] != v[11] != v[14] != v[16] != v[19] != v[22] != v[24] != v[29] != v[30] != v[32] != v[35] != v[39] != v[42] != v[44] != v[46] != v[48], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[4] != v[6] != v[9] != v[10] != v[15] != v[17] != v[18] != v[23] != v[25] != v[28] != v[31] != v[33] != v[34] != v[38] != v[43] != v[45] != v[47] != v[49], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[5] != v[9] != v[11] != v[13] != v[15] != v[17] != v[19] != v[21] != v[26] != v[29] != v[31] != v[35] != v[37] != v[39] != v[41] != v[48] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[4] != v[8] != v[10] != v[12] != v[14] != v[16] != v[18] != v[20] != v[27] != v[28] != v[30] != v[34] != v[36] != v[38] != v[40] != v[49] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[8] != v[12] != v[15] != v[17] != v[20] != v[22] != v[24] != v[27] != v[29] != v[32] != v[34] != v[36] != v[40] != v[42] != v[45] != v[50], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[9] != v[13] != v[14] != v[16] != v[21] != v[23] != v[25] != v[26] != v[28] != v[33] != v[35] != v[37] != v[41] != v[43] != v[44] != v[51], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[2] != v[52], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[3] != v[53], Variable.Bernoulli(0));

                //880-440-0.5
                //Variable.ConstrainEqual(v[13] != v[35] != v[43] != v[69] != v[100] != v[139] != v[195] != v[202] != v[235] != v[244] != v[273] != v[318] != v[330] != v[373] != v[396] != v[409] != v[432] != v[441] != v[460], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[36] != v[44] != v[70] != v[101] != v[120] != v[196] != v[203] != v[236] != v[245] != v[274] != v[319] != v[331] != v[374] != v[397] != v[410] != v[433] != v[442] != v[461], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[37] != v[45] != v[71] != v[102] != v[121] != v[197] != v[204] != v[237] != v[246] != v[275] != v[300] != v[332] != v[375] != v[398] != v[411] != v[434] != v[443] != v[462], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[38] != v[46] != v[72] != v[103] != v[122] != v[198] != v[205] != v[238] != v[247] != v[276] != v[301] != v[333] != v[376] != v[399] != v[412] != v[435] != v[444] != v[463], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[39] != v[47] != v[73] != v[104] != v[123] != v[199] != v[206] != v[239] != v[248] != v[277] != v[302] != v[334] != v[377] != v[380] != v[413] != v[436] != v[445] != v[464], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[20] != v[48] != v[74] != v[105] != v[124] != v[180] != v[207] != v[220] != v[249] != v[278] != v[303] != v[335] != v[378] != v[381] != v[414] != v[437] != v[446] != v[465], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[21] != v[49] != v[75] != v[106] != v[125] != v[181] != v[208] != v[221] != v[250] != v[279] != v[304] != v[336] != v[379] != v[382] != v[415] != v[438] != v[447] != v[466], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[22] != v[50] != v[76] != v[107] != v[126] != v[182] != v[209] != v[222] != v[251] != v[260] != v[305] != v[337] != v[360] != v[383] != v[416] != v[439] != v[448] != v[467], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[23] != v[51] != v[77] != v[108] != v[127] != v[183] != v[210] != v[223] != v[252] != v[261] != v[306] != v[338] != v[361] != v[384] != v[417] != v[420] != v[449] != v[468], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[24] != v[52] != v[78] != v[109] != v[128] != v[184] != v[211] != v[224] != v[253] != v[262] != v[307] != v[339] != v[362] != v[385] != v[418] != v[421] != v[450] != v[469], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[25] != v[53] != v[79] != v[110] != v[129] != v[185] != v[212] != v[225] != v[254] != v[263] != v[308] != v[320] != v[363] != v[386] != v[419] != v[422] != v[451] != v[470], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[26] != v[54] != v[60] != v[111] != v[130] != v[186] != v[213] != v[226] != v[255] != v[264] != v[309] != v[321] != v[364] != v[387] != v[400] != v[423] != v[452] != v[471], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[27] != v[55] != v[61] != v[112] != v[131] != v[187] != v[214] != v[227] != v[256] != v[265] != v[310] != v[322] != v[365] != v[388] != v[401] != v[424] != v[453] != v[472], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[28] != v[56] != v[62] != v[113] != v[132] != v[188] != v[215] != v[228] != v[257] != v[266] != v[311] != v[323] != v[366] != v[389] != v[402] != v[425] != v[454] != v[473], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[29] != v[57] != v[63] != v[114] != v[133] != v[189] != v[216] != v[229] != v[258] != v[267] != v[312] != v[324] != v[367] != v[390] != v[403] != v[426] != v[455] != v[474], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[30] != v[58] != v[64] != v[115] != v[134] != v[190] != v[217] != v[230] != v[259] != v[268] != v[313] != v[325] != v[368] != v[391] != v[404] != v[427] != v[456] != v[475], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[31] != v[59] != v[65] != v[116] != v[135] != v[191] != v[218] != v[231] != v[240] != v[269] != v[314] != v[326] != v[369] != v[392] != v[405] != v[428] != v[457] != v[476], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[32] != v[40] != v[66] != v[117] != v[136] != v[192] != v[219] != v[232] != v[241] != v[270] != v[315] != v[327] != v[370] != v[393] != v[406] != v[429] != v[458] != v[477], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[33] != v[41] != v[67] != v[118] != v[137] != v[193] != v[200] != v[233] != v[242] != v[271] != v[316] != v[328] != v[371] != v[394] != v[407] != v[430] != v[459] != v[478], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[34] != v[42] != v[68] != v[119] != v[138] != v[194] != v[201] != v[234] != v[243] != v[272] != v[317] != v[329] != v[372] != v[395] != v[408] != v[431] != v[440] != v[479], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[54] != v[67] != v[81] != v[101] != v[153] != v[164] != v[180] != v[229] != v[240] != v[296] != v[306] != v[332] != v[343] != v[380] != v[421] != v[440] != v[460] != v[480], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[55] != v[68] != v[82] != v[102] != v[154] != v[165] != v[181] != v[230] != v[241] != v[297] != v[307] != v[333] != v[344] != v[381] != v[422] != v[441] != v[461] != v[481], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[56] != v[69] != v[83] != v[103] != v[155] != v[166] != v[182] != v[231] != v[242] != v[298] != v[308] != v[334] != v[345] != v[382] != v[423] != v[442] != v[462] != v[482], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[57] != v[70] != v[84] != v[104] != v[156] != v[167] != v[183] != v[232] != v[243] != v[299] != v[309] != v[335] != v[346] != v[383] != v[424] != v[443] != v[463] != v[483], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[58] != v[71] != v[85] != v[105] != v[157] != v[168] != v[184] != v[233] != v[244] != v[280] != v[310] != v[336] != v[347] != v[384] != v[425] != v[444] != v[464] != v[484], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[59] != v[72] != v[86] != v[106] != v[158] != v[169] != v[185] != v[234] != v[245] != v[281] != v[311] != v[337] != v[348] != v[385] != v[426] != v[445] != v[465] != v[485], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[40] != v[73] != v[87] != v[107] != v[159] != v[170] != v[186] != v[235] != v[246] != v[282] != v[312] != v[338] != v[349] != v[386] != v[427] != v[446] != v[466] != v[486], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[41] != v[74] != v[88] != v[108] != v[140] != v[171] != v[187] != v[236] != v[247] != v[283] != v[313] != v[339] != v[350] != v[387] != v[428] != v[447] != v[467] != v[487], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[42] != v[75] != v[89] != v[109] != v[141] != v[172] != v[188] != v[237] != v[248] != v[284] != v[314] != v[320] != v[351] != v[388] != v[429] != v[448] != v[468] != v[488], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[43] != v[76] != v[90] != v[110] != v[142] != v[173] != v[189] != v[238] != v[249] != v[285] != v[315] != v[321] != v[352] != v[389] != v[430] != v[449] != v[469] != v[489], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[44] != v[77] != v[91] != v[111] != v[143] != v[174] != v[190] != v[239] != v[250] != v[286] != v[316] != v[322] != v[353] != v[390] != v[431] != v[450] != v[470] != v[490], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[45] != v[78] != v[92] != v[112] != v[144] != v[175] != v[191] != v[220] != v[251] != v[287] != v[317] != v[323] != v[354] != v[391] != v[432] != v[451] != v[471] != v[491], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[46] != v[79] != v[93] != v[113] != v[145] != v[176] != v[192] != v[221] != v[252] != v[288] != v[318] != v[324] != v[355] != v[392] != v[433] != v[452] != v[472] != v[492], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[47] != v[60] != v[94] != v[114] != v[146] != v[177] != v[193] != v[222] != v[253] != v[289] != v[319] != v[325] != v[356] != v[393] != v[434] != v[453] != v[473] != v[493], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[48] != v[61] != v[95] != v[115] != v[147] != v[178] != v[194] != v[223] != v[254] != v[290] != v[300] != v[326] != v[357] != v[394] != v[435] != v[454] != v[474] != v[494], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[49] != v[62] != v[96] != v[116] != v[148] != v[179] != v[195] != v[224] != v[255] != v[291] != v[301] != v[327] != v[358] != v[395] != v[436] != v[455] != v[475] != v[495], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[50] != v[63] != v[97] != v[117] != v[149] != v[160] != v[196] != v[225] != v[256] != v[292] != v[302] != v[328] != v[359] != v[396] != v[437] != v[456] != v[476] != v[496], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[51] != v[64] != v[98] != v[118] != v[150] != v[161] != v[197] != v[226] != v[257] != v[293] != v[303] != v[329] != v[340] != v[397] != v[438] != v[457] != v[477] != v[497], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[52] != v[65] != v[99] != v[119] != v[151] != v[162] != v[198] != v[227] != v[258] != v[294] != v[304] != v[330] != v[341] != v[398] != v[439] != v[458] != v[478] != v[498], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[53] != v[66] != v[80] != v[100] != v[152] != v[163] != v[199] != v[228] != v[259] != v[295] != v[305] != v[331] != v[342] != v[399] != v[420] != v[459] != v[479] != v[499], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[27] != v[40] != v[80] != v[118] != v[127] != v[142] != v[160] != v[191] != v[206] != v[275] != v[283] != v[301] != v[344] != v[366] != v[390] != v[416] != v[480] != v[500], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[28] != v[41] != v[81] != v[119] != v[128] != v[143] != v[161] != v[192] != v[207] != v[276] != v[284] != v[302] != v[345] != v[367] != v[391] != v[417] != v[481] != v[501], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[29] != v[42] != v[82] != v[100] != v[129] != v[144] != v[162] != v[193] != v[208] != v[277] != v[285] != v[303] != v[346] != v[368] != v[392] != v[418] != v[482] != v[502], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[30] != v[43] != v[83] != v[101] != v[130] != v[145] != v[163] != v[194] != v[209] != v[278] != v[286] != v[304] != v[347] != v[369] != v[393] != v[419] != v[483] != v[503], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[31] != v[44] != v[84] != v[102] != v[131] != v[146] != v[164] != v[195] != v[210] != v[279] != v[287] != v[305] != v[348] != v[370] != v[394] != v[400] != v[484] != v[504], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[32] != v[45] != v[85] != v[103] != v[132] != v[147] != v[165] != v[196] != v[211] != v[260] != v[288] != v[306] != v[349] != v[371] != v[395] != v[401] != v[485] != v[505], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[33] != v[46] != v[86] != v[104] != v[133] != v[148] != v[166] != v[197] != v[212] != v[261] != v[289] != v[307] != v[350] != v[372] != v[396] != v[402] != v[486] != v[506], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[34] != v[47] != v[87] != v[105] != v[134] != v[149] != v[167] != v[198] != v[213] != v[262] != v[290] != v[308] != v[351] != v[373] != v[397] != v[403] != v[487] != v[507], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[35] != v[48] != v[88] != v[106] != v[135] != v[150] != v[168] != v[199] != v[214] != v[263] != v[291] != v[309] != v[352] != v[374] != v[398] != v[404] != v[488] != v[508], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[36] != v[49] != v[89] != v[107] != v[136] != v[151] != v[169] != v[180] != v[215] != v[264] != v[292] != v[310] != v[353] != v[375] != v[399] != v[405] != v[489] != v[509], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[37] != v[50] != v[90] != v[108] != v[137] != v[152] != v[170] != v[181] != v[216] != v[265] != v[293] != v[311] != v[354] != v[376] != v[380] != v[406] != v[490] != v[510], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[38] != v[51] != v[91] != v[109] != v[138] != v[153] != v[171] != v[182] != v[217] != v[266] != v[294] != v[312] != v[355] != v[377] != v[381] != v[407] != v[491] != v[511], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[39] != v[52] != v[92] != v[110] != v[139] != v[154] != v[172] != v[183] != v[218] != v[267] != v[295] != v[313] != v[356] != v[378] != v[382] != v[408] != v[492] != v[512], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[20] != v[53] != v[93] != v[111] != v[120] != v[155] != v[173] != v[184] != v[219] != v[268] != v[296] != v[314] != v[357] != v[379] != v[383] != v[409] != v[493] != v[513], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[21] != v[54] != v[94] != v[112] != v[121] != v[156] != v[174] != v[185] != v[200] != v[269] != v[297] != v[315] != v[358] != v[360] != v[384] != v[410] != v[494] != v[514], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[22] != v[55] != v[95] != v[113] != v[122] != v[157] != v[175] != v[186] != v[201] != v[270] != v[298] != v[316] != v[359] != v[361] != v[385] != v[411] != v[495] != v[515], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[23] != v[56] != v[96] != v[114] != v[123] != v[158] != v[176] != v[187] != v[202] != v[271] != v[299] != v[317] != v[340] != v[362] != v[386] != v[412] != v[496] != v[516], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[24] != v[57] != v[97] != v[115] != v[124] != v[159] != v[177] != v[188] != v[203] != v[272] != v[280] != v[318] != v[341] != v[363] != v[387] != v[413] != v[497] != v[517], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[25] != v[58] != v[98] != v[116] != v[125] != v[140] != v[178] != v[189] != v[204] != v[273] != v[281] != v[319] != v[342] != v[364] != v[388] != v[414] != v[498] != v[518], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[26] != v[59] != v[99] != v[117] != v[126] != v[141] != v[179] != v[190] != v[205] != v[274] != v[282] != v[300] != v[343] != v[365] != v[389] != v[415] != v[499] != v[519], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[28] != v[70] != v[97] != v[121] != v[155] != v[179] != v[201] != v[223] != v[253] != v[264] != v[293] != v[324] != v[348] != v[377] != v[419] != v[428] != v[441] != v[500], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[29] != v[71] != v[98] != v[122] != v[156] != v[160] != v[202] != v[224] != v[254] != v[265] != v[294] != v[325] != v[349] != v[378] != v[400] != v[429] != v[442] != v[501], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[30] != v[72] != v[99] != v[123] != v[157] != v[161] != v[203] != v[225] != v[255] != v[266] != v[295] != v[326] != v[350] != v[379] != v[401] != v[430] != v[443] != v[502], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[31] != v[73] != v[80] != v[124] != v[158] != v[162] != v[204] != v[226] != v[256] != v[267] != v[296] != v[327] != v[351] != v[360] != v[402] != v[431] != v[444] != v[503], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[32] != v[74] != v[81] != v[125] != v[159] != v[163] != v[205] != v[227] != v[257] != v[268] != v[297] != v[328] != v[352] != v[361] != v[403] != v[432] != v[445] != v[504], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[33] != v[75] != v[82] != v[126] != v[140] != v[164] != v[206] != v[228] != v[258] != v[269] != v[298] != v[329] != v[353] != v[362] != v[404] != v[433] != v[446] != v[505], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[34] != v[76] != v[83] != v[127] != v[141] != v[165] != v[207] != v[229] != v[259] != v[270] != v[299] != v[330] != v[354] != v[363] != v[405] != v[434] != v[447] != v[506], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[35] != v[77] != v[84] != v[128] != v[142] != v[166] != v[208] != v[230] != v[240] != v[271] != v[280] != v[331] != v[355] != v[364] != v[406] != v[435] != v[448] != v[507], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[36] != v[78] != v[85] != v[129] != v[143] != v[167] != v[209] != v[231] != v[241] != v[272] != v[281] != v[332] != v[356] != v[365] != v[407] != v[436] != v[449] != v[508], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[37] != v[79] != v[86] != v[130] != v[144] != v[168] != v[210] != v[232] != v[242] != v[273] != v[282] != v[333] != v[357] != v[366] != v[408] != v[437] != v[450] != v[509], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[38] != v[60] != v[87] != v[131] != v[145] != v[169] != v[211] != v[233] != v[243] != v[274] != v[283] != v[334] != v[358] != v[367] != v[409] != v[438] != v[451] != v[510], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[39] != v[61] != v[88] != v[132] != v[146] != v[170] != v[212] != v[234] != v[244] != v[275] != v[284] != v[335] != v[359] != v[368] != v[410] != v[439] != v[452] != v[511], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[20] != v[62] != v[89] != v[133] != v[147] != v[171] != v[213] != v[235] != v[245] != v[276] != v[285] != v[336] != v[340] != v[369] != v[411] != v[420] != v[453] != v[512], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[21] != v[63] != v[90] != v[134] != v[148] != v[172] != v[214] != v[236] != v[246] != v[277] != v[286] != v[337] != v[341] != v[370] != v[412] != v[421] != v[454] != v[513], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[22] != v[64] != v[91] != v[135] != v[149] != v[173] != v[215] != v[237] != v[247] != v[278] != v[287] != v[338] != v[342] != v[371] != v[413] != v[422] != v[455] != v[514], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[23] != v[65] != v[92] != v[136] != v[150] != v[174] != v[216] != v[238] != v[248] != v[279] != v[288] != v[339] != v[343] != v[372] != v[414] != v[423] != v[456] != v[515], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[24] != v[66] != v[93] != v[137] != v[151] != v[175] != v[217] != v[239] != v[249] != v[260] != v[289] != v[320] != v[344] != v[373] != v[415] != v[424] != v[457] != v[516], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[25] != v[67] != v[94] != v[138] != v[152] != v[176] != v[218] != v[220] != v[250] != v[261] != v[290] != v[321] != v[345] != v[374] != v[416] != v[425] != v[458] != v[517], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[26] != v[68] != v[95] != v[139] != v[153] != v[177] != v[219] != v[221] != v[251] != v[262] != v[291] != v[322] != v[346] != v[375] != v[417] != v[426] != v[459] != v[518], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[27] != v[69] != v[96] != v[120] != v[154] != v[178] != v[200] != v[222] != v[252] != v[263] != v[292] != v[323] != v[347] != v[376] != v[418] != v[427] != v[440] != v[519], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[25] != v[520], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[26] != v[521], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[27] != v[522], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[28] != v[523], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[29] != v[524], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[30] != v[525], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[31] != v[526], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[32] != v[527], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[33] != v[528], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[34] != v[529], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[35] != v[530], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[36] != v[531], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[37] != v[532], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[38] != v[533], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[39] != v[534], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[20] != v[535], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[21] != v[536], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[22] != v[537], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[23] != v[538], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[24] != v[539], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[32] != v[70] != v[258] != v[321] != v[427] != v[459] != v[540], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[33] != v[71] != v[259] != v[322] != v[428] != v[440] != v[541], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[34] != v[72] != v[240] != v[323] != v[429] != v[441] != v[542], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[35] != v[73] != v[241] != v[324] != v[430] != v[442] != v[543], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[36] != v[74] != v[242] != v[325] != v[431] != v[443] != v[544], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[37] != v[75] != v[243] != v[326] != v[432] != v[444] != v[545], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[38] != v[76] != v[244] != v[327] != v[433] != v[445] != v[546], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[39] != v[77] != v[245] != v[328] != v[434] != v[446] != v[547], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[20] != v[78] != v[246] != v[329] != v[435] != v[447] != v[548], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[21] != v[79] != v[247] != v[330] != v[436] != v[448] != v[549], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[22] != v[60] != v[248] != v[331] != v[437] != v[449] != v[550], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[23] != v[61] != v[249] != v[332] != v[438] != v[450] != v[551], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[24] != v[62] != v[250] != v[333] != v[439] != v[451] != v[552], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[25] != v[63] != v[251] != v[334] != v[420] != v[452] != v[553], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[26] != v[64] != v[252] != v[335] != v[421] != v[453] != v[554], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[27] != v[65] != v[253] != v[336] != v[422] != v[454] != v[555], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[28] != v[66] != v[254] != v[337] != v[423] != v[455] != v[556], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[29] != v[67] != v[255] != v[338] != v[424] != v[456] != v[557], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[30] != v[68] != v[256] != v[339] != v[425] != v[457] != v[558], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[31] != v[69] != v[257] != v[320] != v[426] != v[458] != v[559], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[121] != v[213] != v[233] != v[272] != v[342] != v[378] != v[415] != v[560], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[122] != v[214] != v[234] != v[273] != v[343] != v[379] != v[416] != v[561], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[123] != v[215] != v[235] != v[274] != v[344] != v[360] != v[417] != v[562], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[124] != v[216] != v[236] != v[275] != v[345] != v[361] != v[418] != v[563], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[125] != v[217] != v[237] != v[276] != v[346] != v[362] != v[419] != v[564], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[126] != v[218] != v[238] != v[277] != v[347] != v[363] != v[400] != v[565], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[127] != v[219] != v[239] != v[278] != v[348] != v[364] != v[401] != v[566], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[128] != v[200] != v[220] != v[279] != v[349] != v[365] != v[402] != v[567], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[129] != v[201] != v[221] != v[260] != v[350] != v[366] != v[403] != v[568], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[130] != v[202] != v[222] != v[261] != v[351] != v[367] != v[404] != v[569], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[131] != v[203] != v[223] != v[262] != v[352] != v[368] != v[405] != v[570], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[132] != v[204] != v[224] != v[263] != v[353] != v[369] != v[406] != v[571], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[133] != v[205] != v[225] != v[264] != v[354] != v[370] != v[407] != v[572], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[134] != v[206] != v[226] != v[265] != v[355] != v[371] != v[408] != v[573], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[135] != v[207] != v[227] != v[266] != v[356] != v[372] != v[409] != v[574], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[136] != v[208] != v[228] != v[267] != v[357] != v[373] != v[410] != v[575], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[137] != v[209] != v[229] != v[268] != v[358] != v[374] != v[411] != v[576], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[138] != v[210] != v[230] != v[269] != v[359] != v[375] != v[412] != v[577], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[139] != v[211] != v[231] != v[270] != v[340] != v[376] != v[413] != v[578], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[120] != v[212] != v[232] != v[271] != v[341] != v[377] != v[414] != v[579], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[28] != v[87] != v[150] != v[165] != v[296] != v[580], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[29] != v[88] != v[151] != v[166] != v[297] != v[581], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[30] != v[89] != v[152] != v[167] != v[298] != v[582], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[31] != v[90] != v[153] != v[168] != v[299] != v[583], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[32] != v[91] != v[154] != v[169] != v[280] != v[584], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[33] != v[92] != v[155] != v[170] != v[281] != v[585], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[34] != v[93] != v[156] != v[171] != v[282] != v[586], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[35] != v[94] != v[157] != v[172] != v[283] != v[587], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[36] != v[95] != v[158] != v[173] != v[284] != v[588], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[37] != v[96] != v[159] != v[174] != v[285] != v[589], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[38] != v[97] != v[140] != v[175] != v[286] != v[590], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[39] != v[98] != v[141] != v[176] != v[287] != v[591], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[20] != v[99] != v[142] != v[177] != v[288] != v[592], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[21] != v[80] != v[143] != v[178] != v[289] != v[593], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[22] != v[81] != v[144] != v[179] != v[290] != v[594], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[23] != v[82] != v[145] != v[160] != v[291] != v[595], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[24] != v[83] != v[146] != v[161] != v[292] != v[596], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[25] != v[84] != v[147] != v[162] != v[293] != v[597], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[26] != v[85] != v[148] != v[163] != v[294] != v[598], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[27] != v[86] != v[149] != v[164] != v[295] != v[599], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[33] != v[70] != v[251] != v[336] != v[390] != v[429] != v[445] != v[494] != v[600], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[34] != v[71] != v[252] != v[337] != v[391] != v[430] != v[446] != v[495] != v[601], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[35] != v[72] != v[253] != v[338] != v[392] != v[431] != v[447] != v[496] != v[602], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[36] != v[73] != v[254] != v[339] != v[393] != v[432] != v[448] != v[497] != v[603], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[37] != v[74] != v[255] != v[320] != v[394] != v[433] != v[449] != v[498] != v[604], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[38] != v[75] != v[256] != v[321] != v[395] != v[434] != v[450] != v[499] != v[605], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[39] != v[76] != v[257] != v[322] != v[396] != v[435] != v[451] != v[480] != v[606], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[20] != v[77] != v[258] != v[323] != v[397] != v[436] != v[452] != v[481] != v[607], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[21] != v[78] != v[259] != v[324] != v[398] != v[437] != v[453] != v[482] != v[608], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[22] != v[79] != v[240] != v[325] != v[399] != v[438] != v[454] != v[483] != v[609], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[23] != v[60] != v[241] != v[326] != v[380] != v[439] != v[455] != v[484] != v[610], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[24] != v[61] != v[242] != v[327] != v[381] != v[420] != v[456] != v[485] != v[611], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[25] != v[62] != v[243] != v[328] != v[382] != v[421] != v[457] != v[486] != v[612], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[26] != v[63] != v[244] != v[329] != v[383] != v[422] != v[458] != v[487] != v[613], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[27] != v[64] != v[245] != v[330] != v[384] != v[423] != v[459] != v[488] != v[614], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[28] != v[65] != v[246] != v[331] != v[385] != v[424] != v[440] != v[489] != v[615], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[29] != v[66] != v[247] != v[332] != v[386] != v[425] != v[441] != v[490] != v[616], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[30] != v[67] != v[248] != v[333] != v[387] != v[426] != v[442] != v[491] != v[617], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[31] != v[68] != v[249] != v[334] != v[388] != v[427] != v[443] != v[492] != v[618], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[32] != v[69] != v[250] != v[335] != v[389] != v[428] != v[444] != v[493] != v[619], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[24] != v[216] != v[226] != v[271] != v[347] != v[375] != v[409] != v[620], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[25] != v[217] != v[227] != v[272] != v[348] != v[376] != v[410] != v[621], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[26] != v[218] != v[228] != v[273] != v[349] != v[377] != v[411] != v[622], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[27] != v[219] != v[229] != v[274] != v[350] != v[378] != v[412] != v[623], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[28] != v[200] != v[230] != v[275] != v[351] != v[379] != v[413] != v[624], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[29] != v[201] != v[231] != v[276] != v[352] != v[360] != v[414] != v[625], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[30] != v[202] != v[232] != v[277] != v[353] != v[361] != v[415] != v[626], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[31] != v[203] != v[233] != v[278] != v[354] != v[362] != v[416] != v[627], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[32] != v[204] != v[234] != v[279] != v[355] != v[363] != v[417] != v[628], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[33] != v[205] != v[235] != v[260] != v[356] != v[364] != v[418] != v[629], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[34] != v[206] != v[236] != v[261] != v[357] != v[365] != v[419] != v[630], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[35] != v[207] != v[237] != v[262] != v[358] != v[366] != v[400] != v[631], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[36] != v[208] != v[238] != v[263] != v[359] != v[367] != v[401] != v[632], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[37] != v[209] != v[239] != v[264] != v[340] != v[368] != v[402] != v[633], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[38] != v[210] != v[220] != v[265] != v[341] != v[369] != v[403] != v[634], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[39] != v[211] != v[221] != v[266] != v[342] != v[370] != v[404] != v[635], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[20] != v[212] != v[222] != v[267] != v[343] != v[371] != v[405] != v[636], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[21] != v[213] != v[223] != v[268] != v[344] != v[372] != v[406] != v[637], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[22] != v[214] != v[224] != v[269] != v[345] != v[373] != v[407] != v[638], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[23] != v[215] != v[225] != v[270] != v[346] != v[374] != v[408] != v[639], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[40] != v[91] != v[155] != v[161] != v[291] != v[640], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[41] != v[92] != v[156] != v[162] != v[292] != v[641], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[42] != v[93] != v[157] != v[163] != v[293] != v[642], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[43] != v[94] != v[158] != v[164] != v[294] != v[643], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[44] != v[95] != v[159] != v[165] != v[295] != v[644], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[45] != v[96] != v[140] != v[166] != v[296] != v[645], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[46] != v[97] != v[141] != v[167] != v[297] != v[646], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[47] != v[98] != v[142] != v[168] != v[298] != v[647], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[48] != v[99] != v[143] != v[169] != v[299] != v[648], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[49] != v[80] != v[144] != v[170] != v[280] != v[649], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[50] != v[81] != v[145] != v[171] != v[281] != v[650], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[51] != v[82] != v[146] != v[172] != v[282] != v[651], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[52] != v[83] != v[147] != v[173] != v[283] != v[652], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[53] != v[84] != v[148] != v[174] != v[284] != v[653], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[54] != v[85] != v[149] != v[175] != v[285] != v[654], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[55] != v[86] != v[150] != v[176] != v[286] != v[655], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[56] != v[87] != v[151] != v[177] != v[287] != v[656], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[57] != v[88] != v[152] != v[178] != v[288] != v[657], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[58] != v[89] != v[153] != v[179] != v[289] != v[658], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[59] != v[90] != v[154] != v[160] != v[290] != v[659], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[27] != v[250] != v[329] != v[437] != v[443] != v[460] != v[660], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[28] != v[251] != v[330] != v[438] != v[444] != v[461] != v[661], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[29] != v[252] != v[331] != v[439] != v[445] != v[462] != v[662], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[30] != v[253] != v[332] != v[420] != v[446] != v[463] != v[663], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[31] != v[254] != v[333] != v[421] != v[447] != v[464] != v[664], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[32] != v[255] != v[334] != v[422] != v[448] != v[465] != v[665], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[33] != v[256] != v[335] != v[423] != v[449] != v[466] != v[666], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[34] != v[257] != v[336] != v[424] != v[450] != v[467] != v[667], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[35] != v[258] != v[337] != v[425] != v[451] != v[468] != v[668], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[36] != v[259] != v[338] != v[426] != v[452] != v[469] != v[669], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[37] != v[240] != v[339] != v[427] != v[453] != v[470] != v[670], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[38] != v[241] != v[320] != v[428] != v[454] != v[471] != v[671], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[39] != v[242] != v[321] != v[429] != v[455] != v[472] != v[672], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[20] != v[243] != v[322] != v[430] != v[456] != v[473] != v[673], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[21] != v[244] != v[323] != v[431] != v[457] != v[474] != v[674], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[22] != v[245] != v[324] != v[432] != v[458] != v[475] != v[675], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[23] != v[246] != v[325] != v[433] != v[459] != v[476] != v[676], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[24] != v[247] != v[326] != v[434] != v[440] != v[477] != v[677], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[25] != v[248] != v[327] != v[435] != v[441] != v[478] != v[678], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[26] != v[249] != v[328] != v[436] != v[442] != v[479] != v[679], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[35] != v[209] != v[228] != v[265] != v[372] != v[680], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[36] != v[210] != v[229] != v[266] != v[373] != v[681], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[37] != v[211] != v[230] != v[267] != v[374] != v[682], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[38] != v[212] != v[231] != v[268] != v[375] != v[683], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[39] != v[213] != v[232] != v[269] != v[376] != v[684], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[20] != v[214] != v[233] != v[270] != v[377] != v[685], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[21] != v[215] != v[234] != v[271] != v[378] != v[686], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[22] != v[216] != v[235] != v[272] != v[379] != v[687], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[23] != v[217] != v[236] != v[273] != v[360] != v[688], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[24] != v[218] != v[237] != v[274] != v[361] != v[689], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[25] != v[219] != v[238] != v[275] != v[362] != v[690], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[26] != v[200] != v[239] != v[276] != v[363] != v[691], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[27] != v[201] != v[220] != v[277] != v[364] != v[692], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[28] != v[202] != v[221] != v[278] != v[365] != v[693], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[29] != v[203] != v[222] != v[279] != v[366] != v[694], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[30] != v[204] != v[223] != v[260] != v[367] != v[695], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[31] != v[205] != v[224] != v[261] != v[368] != v[696], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[32] != v[206] != v[225] != v[262] != v[369] != v[697], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[33] != v[207] != v[226] != v[263] != v[370] != v[698], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[34] != v[208] != v[227] != v[264] != v[371] != v[699], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[62] != v[143] != v[400] != v[477] != v[700], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[63] != v[144] != v[401] != v[478] != v[701], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[64] != v[145] != v[402] != v[479] != v[702], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[65] != v[146] != v[403] != v[460] != v[703], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[66] != v[147] != v[404] != v[461] != v[704], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[67] != v[148] != v[405] != v[462] != v[705], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[68] != v[149] != v[406] != v[463] != v[706], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[69] != v[150] != v[407] != v[464] != v[707], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[70] != v[151] != v[408] != v[465] != v[708], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[71] != v[152] != v[409] != v[466] != v[709], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[72] != v[153] != v[410] != v[467] != v[710], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[73] != v[154] != v[411] != v[468] != v[711], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[74] != v[155] != v[412] != v[469] != v[712], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[75] != v[156] != v[413] != v[470] != v[713], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[76] != v[157] != v[414] != v[471] != v[714], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[77] != v[158] != v[415] != v[472] != v[715], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[78] != v[159] != v[416] != v[473] != v[716], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[79] != v[140] != v[417] != v[474] != v[717], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[60] != v[141] != v[418] != v[475] != v[718], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[61] != v[142] != v[419] != v[476] != v[719], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[259] != v[314] != v[321] != v[359] != v[438] != v[720], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[240] != v[315] != v[322] != v[340] != v[439] != v[721], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[241] != v[316] != v[323] != v[341] != v[420] != v[722], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[242] != v[317] != v[324] != v[342] != v[421] != v[723], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[243] != v[318] != v[325] != v[343] != v[422] != v[724], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[244] != v[319] != v[326] != v[344] != v[423] != v[725], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[245] != v[300] != v[327] != v[345] != v[424] != v[726], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[246] != v[301] != v[328] != v[346] != v[425] != v[727], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[247] != v[302] != v[329] != v[347] != v[426] != v[728], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[248] != v[303] != v[330] != v[348] != v[427] != v[729], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[249] != v[304] != v[331] != v[349] != v[428] != v[730], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[250] != v[305] != v[332] != v[350] != v[429] != v[731], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[251] != v[306] != v[333] != v[351] != v[430] != v[732], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[252] != v[307] != v[334] != v[352] != v[431] != v[733], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[253] != v[308] != v[335] != v[353] != v[432] != v[734], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[254] != v[309] != v[336] != v[354] != v[433] != v[735], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[255] != v[310] != v[337] != v[355] != v[434] != v[736], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[256] != v[311] != v[338] != v[356] != v[435] != v[737], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[257] != v[312] != v[339] != v[357] != v[436] != v[738], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[258] != v[313] != v[320] != v[358] != v[437] != v[739], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[30] != v[200] != v[270] != v[364] != v[511] != v[740], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[31] != v[201] != v[271] != v[365] != v[512] != v[741], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[32] != v[202] != v[272] != v[366] != v[513] != v[742], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[33] != v[203] != v[273] != v[367] != v[514] != v[743], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[34] != v[204] != v[274] != v[368] != v[515] != v[744], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[35] != v[205] != v[275] != v[369] != v[516] != v[745], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[36] != v[206] != v[276] != v[370] != v[517] != v[746], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[37] != v[207] != v[277] != v[371] != v[518] != v[747], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[38] != v[208] != v[278] != v[372] != v[519] != v[748], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[39] != v[209] != v[279] != v[373] != v[500] != v[749], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[20] != v[210] != v[260] != v[374] != v[501] != v[750], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[21] != v[211] != v[261] != v[375] != v[502] != v[751], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[22] != v[212] != v[262] != v[376] != v[503] != v[752], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[23] != v[213] != v[263] != v[377] != v[504] != v[753], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[24] != v[214] != v[264] != v[378] != v[505] != v[754], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[25] != v[215] != v[265] != v[379] != v[506] != v[755], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[26] != v[216] != v[266] != v[360] != v[507] != v[756], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[27] != v[217] != v[267] != v[361] != v[508] != v[757], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[28] != v[218] != v[268] != v[362] != v[509] != v[758], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[29] != v[219] != v[269] != v[363] != v[510] != v[759], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[60] != v[225] != v[415] != v[440] != v[760], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[61] != v[226] != v[416] != v[441] != v[761], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[62] != v[227] != v[417] != v[442] != v[762], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[63] != v[228] != v[418] != v[443] != v[763], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[64] != v[229] != v[419] != v[444] != v[764], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[65] != v[230] != v[400] != v[445] != v[765], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[66] != v[231] != v[401] != v[446] != v[766], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[67] != v[232] != v[402] != v[447] != v[767], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[68] != v[233] != v[403] != v[448] != v[768], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[69] != v[234] != v[404] != v[449] != v[769], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[70] != v[235] != v[405] != v[450] != v[770], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[71] != v[236] != v[406] != v[451] != v[771], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[72] != v[237] != v[407] != v[452] != v[772], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[73] != v[238] != v[408] != v[453] != v[773], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[74] != v[239] != v[409] != v[454] != v[774], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[75] != v[220] != v[410] != v[455] != v[775], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[76] != v[221] != v[411] != v[456] != v[776], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[77] != v[222] != v[412] != v[457] != v[777], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[78] != v[223] != v[413] != v[458] != v[778], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[79] != v[224] != v[414] != v[459] != v[779], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[287] != v[328] != v[351] != v[428] != v[780], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[288] != v[329] != v[352] != v[429] != v[781], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[289] != v[330] != v[353] != v[430] != v[782], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[290] != v[331] != v[354] != v[431] != v[783], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[291] != v[332] != v[355] != v[432] != v[784], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[292] != v[333] != v[356] != v[433] != v[785], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[293] != v[334] != v[357] != v[434] != v[786], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[294] != v[335] != v[358] != v[435] != v[787], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[295] != v[336] != v[359] != v[436] != v[788], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[296] != v[337] != v[340] != v[437] != v[789], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[297] != v[338] != v[341] != v[438] != v[790], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[298] != v[339] != v[342] != v[439] != v[791], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[299] != v[320] != v[343] != v[420] != v[792], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[280] != v[321] != v[344] != v[421] != v[793], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[281] != v[322] != v[345] != v[422] != v[794], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[282] != v[323] != v[346] != v[423] != v[795], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[283] != v[324] != v[347] != v[424] != v[796], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[284] != v[325] != v[348] != v[425] != v[797], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[285] != v[326] != v[349] != v[426] != v[798], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[286] != v[327] != v[350] != v[427] != v[799], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[254] != v[271] != v[361] != v[395] != v[800], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[255] != v[272] != v[362] != v[396] != v[801], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[256] != v[273] != v[363] != v[397] != v[802], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[257] != v[274] != v[364] != v[398] != v[803], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[258] != v[275] != v[365] != v[399] != v[804], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[259] != v[276] != v[366] != v[380] != v[805], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[240] != v[277] != v[367] != v[381] != v[806], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[241] != v[278] != v[368] != v[382] != v[807], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[242] != v[279] != v[369] != v[383] != v[808], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[243] != v[260] != v[370] != v[384] != v[809], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[244] != v[261] != v[371] != v[385] != v[810], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[245] != v[262] != v[372] != v[386] != v[811], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[246] != v[263] != v[373] != v[387] != v[812], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[247] != v[264] != v[374] != v[388] != v[813], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[248] != v[265] != v[375] != v[389] != v[814], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[249] != v[266] != v[376] != v[390] != v[815], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[250] != v[267] != v[377] != v[391] != v[816], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[251] != v[268] != v[378] != v[392] != v[817], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[252] != v[269] != v[379] != v[393] != v[818], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[253] != v[270] != v[360] != v[394] != v[819], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[21] != v[141] != v[170] != v[201] != v[820], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[22] != v[142] != v[171] != v[202] != v[821], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[23] != v[143] != v[172] != v[203] != v[822], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[24] != v[144] != v[173] != v[204] != v[823], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[25] != v[145] != v[174] != v[205] != v[824], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[26] != v[146] != v[175] != v[206] != v[825], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[27] != v[147] != v[176] != v[207] != v[826], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[28] != v[148] != v[177] != v[208] != v[827], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[29] != v[149] != v[178] != v[209] != v[828], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[30] != v[150] != v[179] != v[210] != v[829], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[31] != v[151] != v[160] != v[211] != v[830], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[32] != v[152] != v[161] != v[212] != v[831], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[33] != v[153] != v[162] != v[213] != v[832], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[34] != v[154] != v[163] != v[214] != v[833], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[35] != v[155] != v[164] != v[215] != v[834], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[36] != v[156] != v[165] != v[216] != v[835], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[37] != v[157] != v[166] != v[217] != v[836], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[38] != v[158] != v[167] != v[218] != v[837], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[39] != v[159] != v[168] != v[219] != v[838], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[20] != v[140] != v[169] != v[200] != v[839], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[62] != v[180] != v[230] != v[450] != v[840], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[63] != v[181] != v[231] != v[451] != v[841], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[64] != v[182] != v[232] != v[452] != v[842], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[65] != v[183] != v[233] != v[453] != v[843], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[66] != v[184] != v[234] != v[454] != v[844], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[67] != v[185] != v[235] != v[455] != v[845], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[68] != v[186] != v[236] != v[456] != v[846], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[69] != v[187] != v[237] != v[457] != v[847], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[70] != v[188] != v[238] != v[458] != v[848], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[71] != v[189] != v[239] != v[459] != v[849], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[72] != v[190] != v[220] != v[440] != v[850], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[73] != v[191] != v[221] != v[441] != v[851], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[74] != v[192] != v[222] != v[442] != v[852], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[75] != v[193] != v[223] != v[443] != v[853], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[76] != v[194] != v[224] != v[444] != v[854], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[77] != v[195] != v[225] != v[445] != v[855], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[78] != v[196] != v[226] != v[446] != v[856], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[79] != v[197] != v[227] != v[447] != v[857], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[60] != v[198] != v[228] != v[448] != v[858], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[61] != v[199] != v[229] != v[449] != v[859], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[115] != v[332] != v[403] != v[423] != v[860], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[116] != v[333] != v[404] != v[424] != v[861], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[117] != v[334] != v[405] != v[425] != v[862], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[118] != v[335] != v[406] != v[426] != v[863], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[119] != v[336] != v[407] != v[427] != v[864], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[100] != v[337] != v[408] != v[428] != v[865], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[101] != v[338] != v[409] != v[429] != v[866], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[102] != v[339] != v[410] != v[430] != v[867], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[103] != v[320] != v[411] != v[431] != v[868], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[104] != v[321] != v[412] != v[432] != v[869], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[105] != v[322] != v[413] != v[433] != v[870], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[106] != v[323] != v[414] != v[434] != v[871], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[107] != v[324] != v[415] != v[435] != v[872], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[108] != v[325] != v[416] != v[436] != v[873], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[109] != v[326] != v[417] != v[437] != v[874], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[110] != v[327] != v[418] != v[438] != v[875], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[111] != v[328] != v[419] != v[439] != v[876], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[112] != v[329] != v[400] != v[420] != v[877], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[113] != v[330] != v[401] != v[421] != v[878], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[114] != v[331] != v[402] != v[422] != v[879], Variable.Bernoulli(0));

                //1240-440-0.35
                //Variable.ConstrainEqual(v[13] != v[35] != v[43] != v[69] != v[100] != v[139] != v[195] != v[202] != v[235] != v[244] != v[273] != v[318] != v[330] != v[373] != v[396] != v[409] != v[432] != v[441] != v[460], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[36] != v[44] != v[70] != v[101] != v[120] != v[196] != v[203] != v[236] != v[245] != v[274] != v[319] != v[331] != v[374] != v[397] != v[410] != v[433] != v[442] != v[461], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[37] != v[45] != v[71] != v[102] != v[121] != v[197] != v[204] != v[237] != v[246] != v[275] != v[300] != v[332] != v[375] != v[398] != v[411] != v[434] != v[443] != v[462], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[38] != v[46] != v[72] != v[103] != v[122] != v[198] != v[205] != v[238] != v[247] != v[276] != v[301] != v[333] != v[376] != v[399] != v[412] != v[435] != v[444] != v[463], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[39] != v[47] != v[73] != v[104] != v[123] != v[199] != v[206] != v[239] != v[248] != v[277] != v[302] != v[334] != v[377] != v[380] != v[413] != v[436] != v[445] != v[464], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[20] != v[48] != v[74] != v[105] != v[124] != v[180] != v[207] != v[220] != v[249] != v[278] != v[303] != v[335] != v[378] != v[381] != v[414] != v[437] != v[446] != v[465], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[21] != v[49] != v[75] != v[106] != v[125] != v[181] != v[208] != v[221] != v[250] != v[279] != v[304] != v[336] != v[379] != v[382] != v[415] != v[438] != v[447] != v[466], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[22] != v[50] != v[76] != v[107] != v[126] != v[182] != v[209] != v[222] != v[251] != v[260] != v[305] != v[337] != v[360] != v[383] != v[416] != v[439] != v[448] != v[467], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[23] != v[51] != v[77] != v[108] != v[127] != v[183] != v[210] != v[223] != v[252] != v[261] != v[306] != v[338] != v[361] != v[384] != v[417] != v[420] != v[449] != v[468], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[24] != v[52] != v[78] != v[109] != v[128] != v[184] != v[211] != v[224] != v[253] != v[262] != v[307] != v[339] != v[362] != v[385] != v[418] != v[421] != v[450] != v[469], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[25] != v[53] != v[79] != v[110] != v[129] != v[185] != v[212] != v[225] != v[254] != v[263] != v[308] != v[320] != v[363] != v[386] != v[419] != v[422] != v[451] != v[470], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[26] != v[54] != v[60] != v[111] != v[130] != v[186] != v[213] != v[226] != v[255] != v[264] != v[309] != v[321] != v[364] != v[387] != v[400] != v[423] != v[452] != v[471], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[27] != v[55] != v[61] != v[112] != v[131] != v[187] != v[214] != v[227] != v[256] != v[265] != v[310] != v[322] != v[365] != v[388] != v[401] != v[424] != v[453] != v[472], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[28] != v[56] != v[62] != v[113] != v[132] != v[188] != v[215] != v[228] != v[257] != v[266] != v[311] != v[323] != v[366] != v[389] != v[402] != v[425] != v[454] != v[473], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[29] != v[57] != v[63] != v[114] != v[133] != v[189] != v[216] != v[229] != v[258] != v[267] != v[312] != v[324] != v[367] != v[390] != v[403] != v[426] != v[455] != v[474], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[30] != v[58] != v[64] != v[115] != v[134] != v[190] != v[217] != v[230] != v[259] != v[268] != v[313] != v[325] != v[368] != v[391] != v[404] != v[427] != v[456] != v[475], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[31] != v[59] != v[65] != v[116] != v[135] != v[191] != v[218] != v[231] != v[240] != v[269] != v[314] != v[326] != v[369] != v[392] != v[405] != v[428] != v[457] != v[476], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[32] != v[40] != v[66] != v[117] != v[136] != v[192] != v[219] != v[232] != v[241] != v[270] != v[315] != v[327] != v[370] != v[393] != v[406] != v[429] != v[458] != v[477], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[33] != v[41] != v[67] != v[118] != v[137] != v[193] != v[200] != v[233] != v[242] != v[271] != v[316] != v[328] != v[371] != v[394] != v[407] != v[430] != v[459] != v[478], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[34] != v[42] != v[68] != v[119] != v[138] != v[194] != v[201] != v[234] != v[243] != v[272] != v[317] != v[329] != v[372] != v[395] != v[408] != v[431] != v[440] != v[479], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[54] != v[67] != v[81] != v[101] != v[153] != v[164] != v[180] != v[229] != v[240] != v[296] != v[306] != v[332] != v[343] != v[380] != v[421] != v[440] != v[460] != v[480], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[55] != v[68] != v[82] != v[102] != v[154] != v[165] != v[181] != v[230] != v[241] != v[297] != v[307] != v[333] != v[344] != v[381] != v[422] != v[441] != v[461] != v[481], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[56] != v[69] != v[83] != v[103] != v[155] != v[166] != v[182] != v[231] != v[242] != v[298] != v[308] != v[334] != v[345] != v[382] != v[423] != v[442] != v[462] != v[482], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[57] != v[70] != v[84] != v[104] != v[156] != v[167] != v[183] != v[232] != v[243] != v[299] != v[309] != v[335] != v[346] != v[383] != v[424] != v[443] != v[463] != v[483], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[58] != v[71] != v[85] != v[105] != v[157] != v[168] != v[184] != v[233] != v[244] != v[280] != v[310] != v[336] != v[347] != v[384] != v[425] != v[444] != v[464] != v[484], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[59] != v[72] != v[86] != v[106] != v[158] != v[169] != v[185] != v[234] != v[245] != v[281] != v[311] != v[337] != v[348] != v[385] != v[426] != v[445] != v[465] != v[485], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[40] != v[73] != v[87] != v[107] != v[159] != v[170] != v[186] != v[235] != v[246] != v[282] != v[312] != v[338] != v[349] != v[386] != v[427] != v[446] != v[466] != v[486], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[41] != v[74] != v[88] != v[108] != v[140] != v[171] != v[187] != v[236] != v[247] != v[283] != v[313] != v[339] != v[350] != v[387] != v[428] != v[447] != v[467] != v[487], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[42] != v[75] != v[89] != v[109] != v[141] != v[172] != v[188] != v[237] != v[248] != v[284] != v[314] != v[320] != v[351] != v[388] != v[429] != v[448] != v[468] != v[488], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[43] != v[76] != v[90] != v[110] != v[142] != v[173] != v[189] != v[238] != v[249] != v[285] != v[315] != v[321] != v[352] != v[389] != v[430] != v[449] != v[469] != v[489], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[44] != v[77] != v[91] != v[111] != v[143] != v[174] != v[190] != v[239] != v[250] != v[286] != v[316] != v[322] != v[353] != v[390] != v[431] != v[450] != v[470] != v[490], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[45] != v[78] != v[92] != v[112] != v[144] != v[175] != v[191] != v[220] != v[251] != v[287] != v[317] != v[323] != v[354] != v[391] != v[432] != v[451] != v[471] != v[491], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[46] != v[79] != v[93] != v[113] != v[145] != v[176] != v[192] != v[221] != v[252] != v[288] != v[318] != v[324] != v[355] != v[392] != v[433] != v[452] != v[472] != v[492], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[47] != v[60] != v[94] != v[114] != v[146] != v[177] != v[193] != v[222] != v[253] != v[289] != v[319] != v[325] != v[356] != v[393] != v[434] != v[453] != v[473] != v[493], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[48] != v[61] != v[95] != v[115] != v[147] != v[178] != v[194] != v[223] != v[254] != v[290] != v[300] != v[326] != v[357] != v[394] != v[435] != v[454] != v[474] != v[494], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[49] != v[62] != v[96] != v[116] != v[148] != v[179] != v[195] != v[224] != v[255] != v[291] != v[301] != v[327] != v[358] != v[395] != v[436] != v[455] != v[475] != v[495], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[50] != v[63] != v[97] != v[117] != v[149] != v[160] != v[196] != v[225] != v[256] != v[292] != v[302] != v[328] != v[359] != v[396] != v[437] != v[456] != v[476] != v[496], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[51] != v[64] != v[98] != v[118] != v[150] != v[161] != v[197] != v[226] != v[257] != v[293] != v[303] != v[329] != v[340] != v[397] != v[438] != v[457] != v[477] != v[497], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[52] != v[65] != v[99] != v[119] != v[151] != v[162] != v[198] != v[227] != v[258] != v[294] != v[304] != v[330] != v[341] != v[398] != v[439] != v[458] != v[478] != v[498], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[53] != v[66] != v[80] != v[100] != v[152] != v[163] != v[199] != v[228] != v[259] != v[295] != v[305] != v[331] != v[342] != v[399] != v[420] != v[459] != v[479] != v[499], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[27] != v[40] != v[80] != v[118] != v[127] != v[142] != v[160] != v[191] != v[206] != v[275] != v[283] != v[301] != v[344] != v[366] != v[390] != v[416] != v[480] != v[500], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[28] != v[41] != v[81] != v[119] != v[128] != v[143] != v[161] != v[192] != v[207] != v[276] != v[284] != v[302] != v[345] != v[367] != v[391] != v[417] != v[481] != v[501], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[29] != v[42] != v[82] != v[100] != v[129] != v[144] != v[162] != v[193] != v[208] != v[277] != v[285] != v[303] != v[346] != v[368] != v[392] != v[418] != v[482] != v[502], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[30] != v[43] != v[83] != v[101] != v[130] != v[145] != v[163] != v[194] != v[209] != v[278] != v[286] != v[304] != v[347] != v[369] != v[393] != v[419] != v[483] != v[503], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[31] != v[44] != v[84] != v[102] != v[131] != v[146] != v[164] != v[195] != v[210] != v[279] != v[287] != v[305] != v[348] != v[370] != v[394] != v[400] != v[484] != v[504], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[32] != v[45] != v[85] != v[103] != v[132] != v[147] != v[165] != v[196] != v[211] != v[260] != v[288] != v[306] != v[349] != v[371] != v[395] != v[401] != v[485] != v[505], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[33] != v[46] != v[86] != v[104] != v[133] != v[148] != v[166] != v[197] != v[212] != v[261] != v[289] != v[307] != v[350] != v[372] != v[396] != v[402] != v[486] != v[506], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[34] != v[47] != v[87] != v[105] != v[134] != v[149] != v[167] != v[198] != v[213] != v[262] != v[290] != v[308] != v[351] != v[373] != v[397] != v[403] != v[487] != v[507], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[35] != v[48] != v[88] != v[106] != v[135] != v[150] != v[168] != v[199] != v[214] != v[263] != v[291] != v[309] != v[352] != v[374] != v[398] != v[404] != v[488] != v[508], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[36] != v[49] != v[89] != v[107] != v[136] != v[151] != v[169] != v[180] != v[215] != v[264] != v[292] != v[310] != v[353] != v[375] != v[399] != v[405] != v[489] != v[509], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[37] != v[50] != v[90] != v[108] != v[137] != v[152] != v[170] != v[181] != v[216] != v[265] != v[293] != v[311] != v[354] != v[376] != v[380] != v[406] != v[490] != v[510], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[38] != v[51] != v[91] != v[109] != v[138] != v[153] != v[171] != v[182] != v[217] != v[266] != v[294] != v[312] != v[355] != v[377] != v[381] != v[407] != v[491] != v[511], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[39] != v[52] != v[92] != v[110] != v[139] != v[154] != v[172] != v[183] != v[218] != v[267] != v[295] != v[313] != v[356] != v[378] != v[382] != v[408] != v[492] != v[512], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[20] != v[53] != v[93] != v[111] != v[120] != v[155] != v[173] != v[184] != v[219] != v[268] != v[296] != v[314] != v[357] != v[379] != v[383] != v[409] != v[493] != v[513], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[21] != v[54] != v[94] != v[112] != v[121] != v[156] != v[174] != v[185] != v[200] != v[269] != v[297] != v[315] != v[358] != v[360] != v[384] != v[410] != v[494] != v[514], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[22] != v[55] != v[95] != v[113] != v[122] != v[157] != v[175] != v[186] != v[201] != v[270] != v[298] != v[316] != v[359] != v[361] != v[385] != v[411] != v[495] != v[515], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[23] != v[56] != v[96] != v[114] != v[123] != v[158] != v[176] != v[187] != v[202] != v[271] != v[299] != v[317] != v[340] != v[362] != v[386] != v[412] != v[496] != v[516], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[24] != v[57] != v[97] != v[115] != v[124] != v[159] != v[177] != v[188] != v[203] != v[272] != v[280] != v[318] != v[341] != v[363] != v[387] != v[413] != v[497] != v[517], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[25] != v[58] != v[98] != v[116] != v[125] != v[140] != v[178] != v[189] != v[204] != v[273] != v[281] != v[319] != v[342] != v[364] != v[388] != v[414] != v[498] != v[518], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[26] != v[59] != v[99] != v[117] != v[126] != v[141] != v[179] != v[190] != v[205] != v[274] != v[282] != v[300] != v[343] != v[365] != v[389] != v[415] != v[499] != v[519], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[28] != v[70] != v[97] != v[121] != v[155] != v[179] != v[201] != v[223] != v[253] != v[264] != v[293] != v[324] != v[348] != v[377] != v[419] != v[428] != v[441] != v[500], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[29] != v[71] != v[98] != v[122] != v[156] != v[160] != v[202] != v[224] != v[254] != v[265] != v[294] != v[325] != v[349] != v[378] != v[400] != v[429] != v[442] != v[501], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[30] != v[72] != v[99] != v[123] != v[157] != v[161] != v[203] != v[225] != v[255] != v[266] != v[295] != v[326] != v[350] != v[379] != v[401] != v[430] != v[443] != v[502], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[31] != v[73] != v[80] != v[124] != v[158] != v[162] != v[204] != v[226] != v[256] != v[267] != v[296] != v[327] != v[351] != v[360] != v[402] != v[431] != v[444] != v[503], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[32] != v[74] != v[81] != v[125] != v[159] != v[163] != v[205] != v[227] != v[257] != v[268] != v[297] != v[328] != v[352] != v[361] != v[403] != v[432] != v[445] != v[504], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[33] != v[75] != v[82] != v[126] != v[140] != v[164] != v[206] != v[228] != v[258] != v[269] != v[298] != v[329] != v[353] != v[362] != v[404] != v[433] != v[446] != v[505], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[34] != v[76] != v[83] != v[127] != v[141] != v[165] != v[207] != v[229] != v[259] != v[270] != v[299] != v[330] != v[354] != v[363] != v[405] != v[434] != v[447] != v[506], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[35] != v[77] != v[84] != v[128] != v[142] != v[166] != v[208] != v[230] != v[240] != v[271] != v[280] != v[331] != v[355] != v[364] != v[406] != v[435] != v[448] != v[507], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[36] != v[78] != v[85] != v[129] != v[143] != v[167] != v[209] != v[231] != v[241] != v[272] != v[281] != v[332] != v[356] != v[365] != v[407] != v[436] != v[449] != v[508], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[37] != v[79] != v[86] != v[130] != v[144] != v[168] != v[210] != v[232] != v[242] != v[273] != v[282] != v[333] != v[357] != v[366] != v[408] != v[437] != v[450] != v[509], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[38] != v[60] != v[87] != v[131] != v[145] != v[169] != v[211] != v[233] != v[243] != v[274] != v[283] != v[334] != v[358] != v[367] != v[409] != v[438] != v[451] != v[510], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[39] != v[61] != v[88] != v[132] != v[146] != v[170] != v[212] != v[234] != v[244] != v[275] != v[284] != v[335] != v[359] != v[368] != v[410] != v[439] != v[452] != v[511], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[20] != v[62] != v[89] != v[133] != v[147] != v[171] != v[213] != v[235] != v[245] != v[276] != v[285] != v[336] != v[340] != v[369] != v[411] != v[420] != v[453] != v[512], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[21] != v[63] != v[90] != v[134] != v[148] != v[172] != v[214] != v[236] != v[246] != v[277] != v[286] != v[337] != v[341] != v[370] != v[412] != v[421] != v[454] != v[513], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[22] != v[64] != v[91] != v[135] != v[149] != v[173] != v[215] != v[237] != v[247] != v[278] != v[287] != v[338] != v[342] != v[371] != v[413] != v[422] != v[455] != v[514], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[23] != v[65] != v[92] != v[136] != v[150] != v[174] != v[216] != v[238] != v[248] != v[279] != v[288] != v[339] != v[343] != v[372] != v[414] != v[423] != v[456] != v[515], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[24] != v[66] != v[93] != v[137] != v[151] != v[175] != v[217] != v[239] != v[249] != v[260] != v[289] != v[320] != v[344] != v[373] != v[415] != v[424] != v[457] != v[516], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[25] != v[67] != v[94] != v[138] != v[152] != v[176] != v[218] != v[220] != v[250] != v[261] != v[290] != v[321] != v[345] != v[374] != v[416] != v[425] != v[458] != v[517], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[26] != v[68] != v[95] != v[139] != v[153] != v[177] != v[219] != v[221] != v[251] != v[262] != v[291] != v[322] != v[346] != v[375] != v[417] != v[426] != v[459] != v[518], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[27] != v[69] != v[96] != v[120] != v[154] != v[178] != v[200] != v[222] != v[252] != v[263] != v[292] != v[323] != v[347] != v[376] != v[418] != v[427] != v[440] != v[519], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[25] != v[520], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[26] != v[521], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[27] != v[522], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[28] != v[523], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[29] != v[524], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[30] != v[525], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[31] != v[526], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[32] != v[527], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[33] != v[528], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[34] != v[529], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[35] != v[530], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[36] != v[531], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[37] != v[532], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[38] != v[533], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[39] != v[534], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[20] != v[535], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[21] != v[536], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[22] != v[537], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[23] != v[538], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[24] != v[539], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[32] != v[70] != v[258] != v[321] != v[427] != v[459] != v[540], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[33] != v[71] != v[259] != v[322] != v[428] != v[440] != v[541], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[34] != v[72] != v[240] != v[323] != v[429] != v[441] != v[542], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[35] != v[73] != v[241] != v[324] != v[430] != v[442] != v[543], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[36] != v[74] != v[242] != v[325] != v[431] != v[443] != v[544], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[37] != v[75] != v[243] != v[326] != v[432] != v[444] != v[545], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[38] != v[76] != v[244] != v[327] != v[433] != v[445] != v[546], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[39] != v[77] != v[245] != v[328] != v[434] != v[446] != v[547], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[20] != v[78] != v[246] != v[329] != v[435] != v[447] != v[548], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[21] != v[79] != v[247] != v[330] != v[436] != v[448] != v[549], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[22] != v[60] != v[248] != v[331] != v[437] != v[449] != v[550], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[23] != v[61] != v[249] != v[332] != v[438] != v[450] != v[551], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[24] != v[62] != v[250] != v[333] != v[439] != v[451] != v[552], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[25] != v[63] != v[251] != v[334] != v[420] != v[452] != v[553], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[26] != v[64] != v[252] != v[335] != v[421] != v[453] != v[554], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[27] != v[65] != v[253] != v[336] != v[422] != v[454] != v[555], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[28] != v[66] != v[254] != v[337] != v[423] != v[455] != v[556], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[29] != v[67] != v[255] != v[338] != v[424] != v[456] != v[557], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[30] != v[68] != v[256] != v[339] != v[425] != v[457] != v[558], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[31] != v[69] != v[257] != v[320] != v[426] != v[458] != v[559], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[121] != v[213] != v[233] != v[272] != v[342] != v[378] != v[415] != v[560], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[122] != v[214] != v[234] != v[273] != v[343] != v[379] != v[416] != v[561], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[123] != v[215] != v[235] != v[274] != v[344] != v[360] != v[417] != v[562], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[124] != v[216] != v[236] != v[275] != v[345] != v[361] != v[418] != v[563], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[125] != v[217] != v[237] != v[276] != v[346] != v[362] != v[419] != v[564], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[126] != v[218] != v[238] != v[277] != v[347] != v[363] != v[400] != v[565], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[127] != v[219] != v[239] != v[278] != v[348] != v[364] != v[401] != v[566], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[128] != v[200] != v[220] != v[279] != v[349] != v[365] != v[402] != v[567], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[129] != v[201] != v[221] != v[260] != v[350] != v[366] != v[403] != v[568], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[130] != v[202] != v[222] != v[261] != v[351] != v[367] != v[404] != v[569], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[131] != v[203] != v[223] != v[262] != v[352] != v[368] != v[405] != v[570], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[132] != v[204] != v[224] != v[263] != v[353] != v[369] != v[406] != v[571], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[133] != v[205] != v[225] != v[264] != v[354] != v[370] != v[407] != v[572], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[134] != v[206] != v[226] != v[265] != v[355] != v[371] != v[408] != v[573], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[135] != v[207] != v[227] != v[266] != v[356] != v[372] != v[409] != v[574], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[136] != v[208] != v[228] != v[267] != v[357] != v[373] != v[410] != v[575], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[137] != v[209] != v[229] != v[268] != v[358] != v[374] != v[411] != v[576], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[138] != v[210] != v[230] != v[269] != v[359] != v[375] != v[412] != v[577], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[139] != v[211] != v[231] != v[270] != v[340] != v[376] != v[413] != v[578], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[120] != v[212] != v[232] != v[271] != v[341] != v[377] != v[414] != v[579], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[28] != v[87] != v[150] != v[165] != v[296] != v[580], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[29] != v[88] != v[151] != v[166] != v[297] != v[581], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[30] != v[89] != v[152] != v[167] != v[298] != v[582], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[31] != v[90] != v[153] != v[168] != v[299] != v[583], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[32] != v[91] != v[154] != v[169] != v[280] != v[584], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[33] != v[92] != v[155] != v[170] != v[281] != v[585], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[34] != v[93] != v[156] != v[171] != v[282] != v[586], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[35] != v[94] != v[157] != v[172] != v[283] != v[587], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[36] != v[95] != v[158] != v[173] != v[284] != v[588], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[37] != v[96] != v[159] != v[174] != v[285] != v[589], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[38] != v[97] != v[140] != v[175] != v[286] != v[590], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[39] != v[98] != v[141] != v[176] != v[287] != v[591], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[20] != v[99] != v[142] != v[177] != v[288] != v[592], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[21] != v[80] != v[143] != v[178] != v[289] != v[593], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[22] != v[81] != v[144] != v[179] != v[290] != v[594], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[23] != v[82] != v[145] != v[160] != v[291] != v[595], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[24] != v[83] != v[146] != v[161] != v[292] != v[596], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[25] != v[84] != v[147] != v[162] != v[293] != v[597], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[26] != v[85] != v[148] != v[163] != v[294] != v[598], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[27] != v[86] != v[149] != v[164] != v[295] != v[599], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[33] != v[70] != v[251] != v[336] != v[390] != v[429] != v[445] != v[494] != v[600], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[34] != v[71] != v[252] != v[337] != v[391] != v[430] != v[446] != v[495] != v[601], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[35] != v[72] != v[253] != v[338] != v[392] != v[431] != v[447] != v[496] != v[602], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[36] != v[73] != v[254] != v[339] != v[393] != v[432] != v[448] != v[497] != v[603], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[37] != v[74] != v[255] != v[320] != v[394] != v[433] != v[449] != v[498] != v[604], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[38] != v[75] != v[256] != v[321] != v[395] != v[434] != v[450] != v[499] != v[605], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[39] != v[76] != v[257] != v[322] != v[396] != v[435] != v[451] != v[480] != v[606], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[20] != v[77] != v[258] != v[323] != v[397] != v[436] != v[452] != v[481] != v[607], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[21] != v[78] != v[259] != v[324] != v[398] != v[437] != v[453] != v[482] != v[608], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[22] != v[79] != v[240] != v[325] != v[399] != v[438] != v[454] != v[483] != v[609], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[23] != v[60] != v[241] != v[326] != v[380] != v[439] != v[455] != v[484] != v[610], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[24] != v[61] != v[242] != v[327] != v[381] != v[420] != v[456] != v[485] != v[611], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[25] != v[62] != v[243] != v[328] != v[382] != v[421] != v[457] != v[486] != v[612], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[26] != v[63] != v[244] != v[329] != v[383] != v[422] != v[458] != v[487] != v[613], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[27] != v[64] != v[245] != v[330] != v[384] != v[423] != v[459] != v[488] != v[614], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[28] != v[65] != v[246] != v[331] != v[385] != v[424] != v[440] != v[489] != v[615], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[29] != v[66] != v[247] != v[332] != v[386] != v[425] != v[441] != v[490] != v[616], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[30] != v[67] != v[248] != v[333] != v[387] != v[426] != v[442] != v[491] != v[617], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[31] != v[68] != v[249] != v[334] != v[388] != v[427] != v[443] != v[492] != v[618], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[32] != v[69] != v[250] != v[335] != v[389] != v[428] != v[444] != v[493] != v[619], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[24] != v[216] != v[226] != v[271] != v[347] != v[375] != v[409] != v[620], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[25] != v[217] != v[227] != v[272] != v[348] != v[376] != v[410] != v[621], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[26] != v[218] != v[228] != v[273] != v[349] != v[377] != v[411] != v[622], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[27] != v[219] != v[229] != v[274] != v[350] != v[378] != v[412] != v[623], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[28] != v[200] != v[230] != v[275] != v[351] != v[379] != v[413] != v[624], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[29] != v[201] != v[231] != v[276] != v[352] != v[360] != v[414] != v[625], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[30] != v[202] != v[232] != v[277] != v[353] != v[361] != v[415] != v[626], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[31] != v[203] != v[233] != v[278] != v[354] != v[362] != v[416] != v[627], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[32] != v[204] != v[234] != v[279] != v[355] != v[363] != v[417] != v[628], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[33] != v[205] != v[235] != v[260] != v[356] != v[364] != v[418] != v[629], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[34] != v[206] != v[236] != v[261] != v[357] != v[365] != v[419] != v[630], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[35] != v[207] != v[237] != v[262] != v[358] != v[366] != v[400] != v[631], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[36] != v[208] != v[238] != v[263] != v[359] != v[367] != v[401] != v[632], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[37] != v[209] != v[239] != v[264] != v[340] != v[368] != v[402] != v[633], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[38] != v[210] != v[220] != v[265] != v[341] != v[369] != v[403] != v[634], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[39] != v[211] != v[221] != v[266] != v[342] != v[370] != v[404] != v[635], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[20] != v[212] != v[222] != v[267] != v[343] != v[371] != v[405] != v[636], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[21] != v[213] != v[223] != v[268] != v[344] != v[372] != v[406] != v[637], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[22] != v[214] != v[224] != v[269] != v[345] != v[373] != v[407] != v[638], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[23] != v[215] != v[225] != v[270] != v[346] != v[374] != v[408] != v[639], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[40] != v[91] != v[155] != v[161] != v[291] != v[640], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[41] != v[92] != v[156] != v[162] != v[292] != v[641], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[42] != v[93] != v[157] != v[163] != v[293] != v[642], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[43] != v[94] != v[158] != v[164] != v[294] != v[643], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[44] != v[95] != v[159] != v[165] != v[295] != v[644], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[45] != v[96] != v[140] != v[166] != v[296] != v[645], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[46] != v[97] != v[141] != v[167] != v[297] != v[646], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[47] != v[98] != v[142] != v[168] != v[298] != v[647], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[48] != v[99] != v[143] != v[169] != v[299] != v[648], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[49] != v[80] != v[144] != v[170] != v[280] != v[649], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[50] != v[81] != v[145] != v[171] != v[281] != v[650], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[51] != v[82] != v[146] != v[172] != v[282] != v[651], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[52] != v[83] != v[147] != v[173] != v[283] != v[652], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[53] != v[84] != v[148] != v[174] != v[284] != v[653], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[54] != v[85] != v[149] != v[175] != v[285] != v[654], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[55] != v[86] != v[150] != v[176] != v[286] != v[655], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[56] != v[87] != v[151] != v[177] != v[287] != v[656], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[57] != v[88] != v[152] != v[178] != v[288] != v[657], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[58] != v[89] != v[153] != v[179] != v[289] != v[658], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[59] != v[90] != v[154] != v[160] != v[290] != v[659], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[27] != v[250] != v[329] != v[437] != v[443] != v[460] != v[660], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[28] != v[251] != v[330] != v[438] != v[444] != v[461] != v[661], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[29] != v[252] != v[331] != v[439] != v[445] != v[462] != v[662], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[30] != v[253] != v[332] != v[420] != v[446] != v[463] != v[663], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[31] != v[254] != v[333] != v[421] != v[447] != v[464] != v[664], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[32] != v[255] != v[334] != v[422] != v[448] != v[465] != v[665], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[33] != v[256] != v[335] != v[423] != v[449] != v[466] != v[666], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[34] != v[257] != v[336] != v[424] != v[450] != v[467] != v[667], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[35] != v[258] != v[337] != v[425] != v[451] != v[468] != v[668], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[36] != v[259] != v[338] != v[426] != v[452] != v[469] != v[669], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[37] != v[240] != v[339] != v[427] != v[453] != v[470] != v[670], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[38] != v[241] != v[320] != v[428] != v[454] != v[471] != v[671], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[39] != v[242] != v[321] != v[429] != v[455] != v[472] != v[672], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[20] != v[243] != v[322] != v[430] != v[456] != v[473] != v[673], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[21] != v[244] != v[323] != v[431] != v[457] != v[474] != v[674], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[22] != v[245] != v[324] != v[432] != v[458] != v[475] != v[675], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[23] != v[246] != v[325] != v[433] != v[459] != v[476] != v[676], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[24] != v[247] != v[326] != v[434] != v[440] != v[477] != v[677], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[25] != v[248] != v[327] != v[435] != v[441] != v[478] != v[678], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[26] != v[249] != v[328] != v[436] != v[442] != v[479] != v[679], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[35] != v[209] != v[228] != v[265] != v[372] != v[680], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[36] != v[210] != v[229] != v[266] != v[373] != v[681], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[37] != v[211] != v[230] != v[267] != v[374] != v[682], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[38] != v[212] != v[231] != v[268] != v[375] != v[683], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[39] != v[213] != v[232] != v[269] != v[376] != v[684], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[20] != v[214] != v[233] != v[270] != v[377] != v[685], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[21] != v[215] != v[234] != v[271] != v[378] != v[686], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[22] != v[216] != v[235] != v[272] != v[379] != v[687], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[23] != v[217] != v[236] != v[273] != v[360] != v[688], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[24] != v[218] != v[237] != v[274] != v[361] != v[689], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[25] != v[219] != v[238] != v[275] != v[362] != v[690], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[26] != v[200] != v[239] != v[276] != v[363] != v[691], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[27] != v[201] != v[220] != v[277] != v[364] != v[692], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[28] != v[202] != v[221] != v[278] != v[365] != v[693], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[29] != v[203] != v[222] != v[279] != v[366] != v[694], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[30] != v[204] != v[223] != v[260] != v[367] != v[695], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[31] != v[205] != v[224] != v[261] != v[368] != v[696], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[32] != v[206] != v[225] != v[262] != v[369] != v[697], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[33] != v[207] != v[226] != v[263] != v[370] != v[698], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[34] != v[208] != v[227] != v[264] != v[371] != v[699], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[62] != v[143] != v[400] != v[477] != v[700], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[63] != v[144] != v[401] != v[478] != v[701], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[64] != v[145] != v[402] != v[479] != v[702], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[65] != v[146] != v[403] != v[460] != v[703], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[66] != v[147] != v[404] != v[461] != v[704], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[67] != v[148] != v[405] != v[462] != v[705], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[68] != v[149] != v[406] != v[463] != v[706], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[69] != v[150] != v[407] != v[464] != v[707], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[70] != v[151] != v[408] != v[465] != v[708], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[71] != v[152] != v[409] != v[466] != v[709], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[72] != v[153] != v[410] != v[467] != v[710], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[73] != v[154] != v[411] != v[468] != v[711], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[74] != v[155] != v[412] != v[469] != v[712], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[75] != v[156] != v[413] != v[470] != v[713], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[76] != v[157] != v[414] != v[471] != v[714], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[77] != v[158] != v[415] != v[472] != v[715], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[78] != v[159] != v[416] != v[473] != v[716], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[79] != v[140] != v[417] != v[474] != v[717], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[60] != v[141] != v[418] != v[475] != v[718], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[61] != v[142] != v[419] != v[476] != v[719], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[259] != v[314] != v[321] != v[359] != v[438] != v[720], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[240] != v[315] != v[322] != v[340] != v[439] != v[721], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[241] != v[316] != v[323] != v[341] != v[420] != v[722], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[242] != v[317] != v[324] != v[342] != v[421] != v[723], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[243] != v[318] != v[325] != v[343] != v[422] != v[724], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[244] != v[319] != v[326] != v[344] != v[423] != v[725], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[245] != v[300] != v[327] != v[345] != v[424] != v[726], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[246] != v[301] != v[328] != v[346] != v[425] != v[727], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[247] != v[302] != v[329] != v[347] != v[426] != v[728], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[248] != v[303] != v[330] != v[348] != v[427] != v[729], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[249] != v[304] != v[331] != v[349] != v[428] != v[730], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[250] != v[305] != v[332] != v[350] != v[429] != v[731], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[251] != v[306] != v[333] != v[351] != v[430] != v[732], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[252] != v[307] != v[334] != v[352] != v[431] != v[733], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[253] != v[308] != v[335] != v[353] != v[432] != v[734], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[254] != v[309] != v[336] != v[354] != v[433] != v[735], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[255] != v[310] != v[337] != v[355] != v[434] != v[736], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[256] != v[311] != v[338] != v[356] != v[435] != v[737], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[257] != v[312] != v[339] != v[357] != v[436] != v[738], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[258] != v[313] != v[320] != v[358] != v[437] != v[739], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[30] != v[200] != v[270] != v[364] != v[511] != v[740], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[31] != v[201] != v[271] != v[365] != v[512] != v[741], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[32] != v[202] != v[272] != v[366] != v[513] != v[742], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[33] != v[203] != v[273] != v[367] != v[514] != v[743], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[34] != v[204] != v[274] != v[368] != v[515] != v[744], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[35] != v[205] != v[275] != v[369] != v[516] != v[745], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[36] != v[206] != v[276] != v[370] != v[517] != v[746], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[37] != v[207] != v[277] != v[371] != v[518] != v[747], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[38] != v[208] != v[278] != v[372] != v[519] != v[748], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[39] != v[209] != v[279] != v[373] != v[500] != v[749], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[20] != v[210] != v[260] != v[374] != v[501] != v[750], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[21] != v[211] != v[261] != v[375] != v[502] != v[751], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[22] != v[212] != v[262] != v[376] != v[503] != v[752], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[23] != v[213] != v[263] != v[377] != v[504] != v[753], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[24] != v[214] != v[264] != v[378] != v[505] != v[754], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[25] != v[215] != v[265] != v[379] != v[506] != v[755], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[26] != v[216] != v[266] != v[360] != v[507] != v[756], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[27] != v[217] != v[267] != v[361] != v[508] != v[757], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[28] != v[218] != v[268] != v[362] != v[509] != v[758], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[29] != v[219] != v[269] != v[363] != v[510] != v[759], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[60] != v[225] != v[415] != v[440] != v[760], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[61] != v[226] != v[416] != v[441] != v[761], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[62] != v[227] != v[417] != v[442] != v[762], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[63] != v[228] != v[418] != v[443] != v[763], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[64] != v[229] != v[419] != v[444] != v[764], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[65] != v[230] != v[400] != v[445] != v[765], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[66] != v[231] != v[401] != v[446] != v[766], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[67] != v[232] != v[402] != v[447] != v[767], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[68] != v[233] != v[403] != v[448] != v[768], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[69] != v[234] != v[404] != v[449] != v[769], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[70] != v[235] != v[405] != v[450] != v[770], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[71] != v[236] != v[406] != v[451] != v[771], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[72] != v[237] != v[407] != v[452] != v[772], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[73] != v[238] != v[408] != v[453] != v[773], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[74] != v[239] != v[409] != v[454] != v[774], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[75] != v[220] != v[410] != v[455] != v[775], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[76] != v[221] != v[411] != v[456] != v[776], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[77] != v[222] != v[412] != v[457] != v[777], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[78] != v[223] != v[413] != v[458] != v[778], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[79] != v[224] != v[414] != v[459] != v[779], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[287] != v[328] != v[351] != v[428] != v[780], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[288] != v[329] != v[352] != v[429] != v[781], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[289] != v[330] != v[353] != v[430] != v[782], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[290] != v[331] != v[354] != v[431] != v[783], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[291] != v[332] != v[355] != v[432] != v[784], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[292] != v[333] != v[356] != v[433] != v[785], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[293] != v[334] != v[357] != v[434] != v[786], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[294] != v[335] != v[358] != v[435] != v[787], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[295] != v[336] != v[359] != v[436] != v[788], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[296] != v[337] != v[340] != v[437] != v[789], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[297] != v[338] != v[341] != v[438] != v[790], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[298] != v[339] != v[342] != v[439] != v[791], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[299] != v[320] != v[343] != v[420] != v[792], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[280] != v[321] != v[344] != v[421] != v[793], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[281] != v[322] != v[345] != v[422] != v[794], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[282] != v[323] != v[346] != v[423] != v[795], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[283] != v[324] != v[347] != v[424] != v[796], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[284] != v[325] != v[348] != v[425] != v[797], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[285] != v[326] != v[349] != v[426] != v[798], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[286] != v[327] != v[350] != v[427] != v[799], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[254] != v[271] != v[361] != v[395] != v[800], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[255] != v[272] != v[362] != v[396] != v[801], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[256] != v[273] != v[363] != v[397] != v[802], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[257] != v[274] != v[364] != v[398] != v[803], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[258] != v[275] != v[365] != v[399] != v[804], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[259] != v[276] != v[366] != v[380] != v[805], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[240] != v[277] != v[367] != v[381] != v[806], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[241] != v[278] != v[368] != v[382] != v[807], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[242] != v[279] != v[369] != v[383] != v[808], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[243] != v[260] != v[370] != v[384] != v[809], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[244] != v[261] != v[371] != v[385] != v[810], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[245] != v[262] != v[372] != v[386] != v[811], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[246] != v[263] != v[373] != v[387] != v[812], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[247] != v[264] != v[374] != v[388] != v[813], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[248] != v[265] != v[375] != v[389] != v[814], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[249] != v[266] != v[376] != v[390] != v[815], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[250] != v[267] != v[377] != v[391] != v[816], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[251] != v[268] != v[378] != v[392] != v[817], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[252] != v[269] != v[379] != v[393] != v[818], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[253] != v[270] != v[360] != v[394] != v[819], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[21] != v[141] != v[170] != v[201] != v[820], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[22] != v[142] != v[171] != v[202] != v[821], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[23] != v[143] != v[172] != v[203] != v[822], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[24] != v[144] != v[173] != v[204] != v[823], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[25] != v[145] != v[174] != v[205] != v[824], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[26] != v[146] != v[175] != v[206] != v[825], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[27] != v[147] != v[176] != v[207] != v[826], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[28] != v[148] != v[177] != v[208] != v[827], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[29] != v[149] != v[178] != v[209] != v[828], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[30] != v[150] != v[179] != v[210] != v[829], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[31] != v[151] != v[160] != v[211] != v[830], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[32] != v[152] != v[161] != v[212] != v[831], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[33] != v[153] != v[162] != v[213] != v[832], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[34] != v[154] != v[163] != v[214] != v[833], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[35] != v[155] != v[164] != v[215] != v[834], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[36] != v[156] != v[165] != v[216] != v[835], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[37] != v[157] != v[166] != v[217] != v[836], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[38] != v[158] != v[167] != v[218] != v[837], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[39] != v[159] != v[168] != v[219] != v[838], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[20] != v[140] != v[169] != v[200] != v[839], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[62] != v[180] != v[230] != v[450] != v[840], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[63] != v[181] != v[231] != v[451] != v[841], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[64] != v[182] != v[232] != v[452] != v[842], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[65] != v[183] != v[233] != v[453] != v[843], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[66] != v[184] != v[234] != v[454] != v[844], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[67] != v[185] != v[235] != v[455] != v[845], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[68] != v[186] != v[236] != v[456] != v[846], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[69] != v[187] != v[237] != v[457] != v[847], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[70] != v[188] != v[238] != v[458] != v[848], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[71] != v[189] != v[239] != v[459] != v[849], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[72] != v[190] != v[220] != v[440] != v[850], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[73] != v[191] != v[221] != v[441] != v[851], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[74] != v[192] != v[222] != v[442] != v[852], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[75] != v[193] != v[223] != v[443] != v[853], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[76] != v[194] != v[224] != v[444] != v[854], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[77] != v[195] != v[225] != v[445] != v[855], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[78] != v[196] != v[226] != v[446] != v[856], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[79] != v[197] != v[227] != v[447] != v[857], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[60] != v[198] != v[228] != v[448] != v[858], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[61] != v[199] != v[229] != v[449] != v[859], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[115] != v[332] != v[403] != v[423] != v[860], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[116] != v[333] != v[404] != v[424] != v[861], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[117] != v[334] != v[405] != v[425] != v[862], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[118] != v[335] != v[406] != v[426] != v[863], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[119] != v[336] != v[407] != v[427] != v[864], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[100] != v[337] != v[408] != v[428] != v[865], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[101] != v[338] != v[409] != v[429] != v[866], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[102] != v[339] != v[410] != v[430] != v[867], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[103] != v[320] != v[411] != v[431] != v[868], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[104] != v[321] != v[412] != v[432] != v[869], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[105] != v[322] != v[413] != v[433] != v[870], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[106] != v[323] != v[414] != v[434] != v[871], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[107] != v[324] != v[415] != v[435] != v[872], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[108] != v[325] != v[416] != v[436] != v[873], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[109] != v[326] != v[417] != v[437] != v[874], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[110] != v[327] != v[418] != v[438] != v[875], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[111] != v[328] != v[419] != v[439] != v[876], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[112] != v[329] != v[400] != v[420] != v[877], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[113] != v[330] != v[401] != v[421] != v[878], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[114] != v[331] != v[402] != v[422] != v[879], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[240] != v[275] != v[356] != v[880], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[241] != v[276] != v[357] != v[881], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[242] != v[277] != v[358] != v[882], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[243] != v[278] != v[359] != v[883], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[244] != v[279] != v[340] != v[884], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[245] != v[260] != v[341] != v[885], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[246] != v[261] != v[342] != v[886], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[247] != v[262] != v[343] != v[887], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[248] != v[263] != v[344] != v[888], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[249] != v[264] != v[345] != v[889], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[250] != v[265] != v[346] != v[890], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[251] != v[266] != v[347] != v[891], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[252] != v[267] != v[348] != v[892], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[253] != v[268] != v[349] != v[893], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[254] != v[269] != v[350] != v[894], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[255] != v[270] != v[351] != v[895], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[256] != v[271] != v[352] != v[896], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[257] != v[272] != v[353] != v[897], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[258] != v[273] != v[354] != v[898], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[259] != v[274] != v[355] != v[899], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[50] != v[213] != v[365] != v[900], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[51] != v[214] != v[366] != v[901], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[52] != v[215] != v[367] != v[902], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[53] != v[216] != v[368] != v[903], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[54] != v[217] != v[369] != v[904], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[55] != v[218] != v[370] != v[905], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[56] != v[219] != v[371] != v[906], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[57] != v[200] != v[372] != v[907], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[58] != v[201] != v[373] != v[908], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[59] != v[202] != v[374] != v[909], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[40] != v[203] != v[375] != v[910], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[41] != v[204] != v[376] != v[911], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[42] != v[205] != v[377] != v[912], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[43] != v[206] != v[378] != v[913], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[44] != v[207] != v[379] != v[914], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[45] != v[208] != v[360] != v[915], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[46] != v[209] != v[361] != v[916], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[47] != v[210] != v[362] != v[917], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[48] != v[211] != v[363] != v[918], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[49] != v[212] != v[364] != v[919], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[60] != v[90] != v[227] != v[441] != v[920], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[61] != v[91] != v[228] != v[442] != v[921], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[62] != v[92] != v[229] != v[443] != v[922], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[63] != v[93] != v[230] != v[444] != v[923], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[64] != v[94] != v[231] != v[445] != v[924], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[65] != v[95] != v[232] != v[446] != v[925], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[66] != v[96] != v[233] != v[447] != v[926], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[67] != v[97] != v[234] != v[448] != v[927], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[68] != v[98] != v[235] != v[449] != v[928], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[69] != v[99] != v[236] != v[450] != v[929], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[70] != v[80] != v[237] != v[451] != v[930], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[71] != v[81] != v[238] != v[452] != v[931], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[72] != v[82] != v[239] != v[453] != v[932], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[73] != v[83] != v[220] != v[454] != v[933], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[74] != v[84] != v[221] != v[455] != v[934], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[75] != v[85] != v[222] != v[456] != v[935], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[76] != v[86] != v[223] != v[457] != v[936], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[77] != v[87] != v[224] != v[458] != v[937], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[78] != v[88] != v[225] != v[459] != v[938], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[79] != v[89] != v[226] != v[440] != v[939], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[135] != v[156] != v[286] != v[940], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[136] != v[157] != v[287] != v[941], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[137] != v[158] != v[288] != v[942], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[138] != v[159] != v[289] != v[943], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[139] != v[140] != v[290] != v[944], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[120] != v[141] != v[291] != v[945], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[121] != v[142] != v[292] != v[946], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[122] != v[143] != v[293] != v[947], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[123] != v[144] != v[294] != v[948], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[124] != v[145] != v[295] != v[949], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[125] != v[146] != v[296] != v[950], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[126] != v[147] != v[297] != v[951], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[127] != v[148] != v[298] != v[952], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[128] != v[149] != v[299] != v[953], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[129] != v[150] != v[280] != v[954], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[130] != v[151] != v[281] != v[955], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[131] != v[152] != v[282] != v[956], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[132] != v[153] != v[283] != v[957], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[133] != v[154] != v[284] != v[958], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[134] != v[155] != v[285] != v[959], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[50] != v[98] != v[307] != v[960], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[51] != v[99] != v[308] != v[961], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[52] != v[80] != v[309] != v[962], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[53] != v[81] != v[310] != v[963], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[54] != v[82] != v[311] != v[964], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[55] != v[83] != v[312] != v[965], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[56] != v[84] != v[313] != v[966], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[57] != v[85] != v[314] != v[967], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[58] != v[86] != v[315] != v[968], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[59] != v[87] != v[316] != v[969], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[40] != v[88] != v[317] != v[970], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[41] != v[89] != v[318] != v[971], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[42] != v[90] != v[319] != v[972], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[43] != v[91] != v[300] != v[973], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[44] != v[92] != v[301] != v[974], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[45] != v[93] != v[302] != v[975], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[46] != v[94] != v[303] != v[976], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[47] != v[95] != v[304] != v[977], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[48] != v[96] != v[305] != v[978], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[49] != v[97] != v[306] != v[979], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[129] != v[164] != v[980], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[130] != v[165] != v[981], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[131] != v[166] != v[982], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[132] != v[167] != v[983], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[133] != v[168] != v[984], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[134] != v[169] != v[985], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[135] != v[170] != v[986], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[136] != v[171] != v[987], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[137] != v[172] != v[988], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[138] != v[173] != v[989], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[139] != v[174] != v[990], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[120] != v[175] != v[991], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[121] != v[176] != v[992], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[122] != v[177] != v[993], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[123] != v[178] != v[994], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[124] != v[179] != v[995], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[125] != v[160] != v[996], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[126] != v[161] != v[997], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[127] != v[162] != v[998], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[128] != v[163] != v[999], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[93] != v[390] != v[434] != v[1000], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[94] != v[391] != v[435] != v[1001], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[95] != v[392] != v[436] != v[1002], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[96] != v[393] != v[437] != v[1003], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[97] != v[394] != v[438] != v[1004], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[98] != v[395] != v[439] != v[1005], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[99] != v[396] != v[420] != v[1006], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[80] != v[397] != v[421] != v[1007], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[81] != v[398] != v[422] != v[1008], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[82] != v[399] != v[423] != v[1009], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[83] != v[380] != v[424] != v[1010], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[84] != v[381] != v[425] != v[1011], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[85] != v[382] != v[426] != v[1012], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[86] != v[383] != v[427] != v[1013], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[87] != v[384] != v[428] != v[1014], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[88] != v[385] != v[429] != v[1015], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[89] != v[386] != v[430] != v[1016], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[90] != v[387] != v[431] != v[1017], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[91] != v[388] != v[432] != v[1018], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[92] != v[389] != v[433] != v[1019], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[288] != v[377] != v[509] != v[1020], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[289] != v[378] != v[510] != v[1021], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[290] != v[379] != v[511] != v[1022], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[291] != v[360] != v[512] != v[1023], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[292] != v[361] != v[513] != v[1024], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[293] != v[362] != v[514] != v[1025], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[294] != v[363] != v[515] != v[1026], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[295] != v[364] != v[516] != v[1027], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[296] != v[365] != v[517] != v[1028], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[297] != v[366] != v[518] != v[1029], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[298] != v[367] != v[519] != v[1030], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[299] != v[368] != v[500] != v[1031], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[280] != v[369] != v[501] != v[1032], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[281] != v[370] != v[502] != v[1033], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[282] != v[371] != v[503] != v[1034], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[283] != v[372] != v[504] != v[1035], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[284] != v[373] != v[505] != v[1036], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[285] != v[374] != v[506] != v[1037], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[286] != v[375] != v[507] != v[1038], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[287] != v[376] != v[508] != v[1039], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[203] != v[265] != v[495] != v[1040], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[204] != v[266] != v[496] != v[1041], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[205] != v[267] != v[497] != v[1042], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[206] != v[268] != v[498] != v[1043], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[207] != v[269] != v[499] != v[1044], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[208] != v[270] != v[480] != v[1045], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[209] != v[271] != v[481] != v[1046], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[210] != v[272] != v[482] != v[1047], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[211] != v[273] != v[483] != v[1048], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[212] != v[274] != v[484] != v[1049], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[213] != v[275] != v[485] != v[1050], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[214] != v[276] != v[486] != v[1051], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[215] != v[277] != v[487] != v[1052], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[216] != v[278] != v[488] != v[1053], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[217] != v[279] != v[489] != v[1054], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[218] != v[260] != v[490] != v[1055], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[219] != v[261] != v[491] != v[1056], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[200] != v[262] != v[492] != v[1057], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[201] != v[263] != v[493] != v[1058], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[202] != v[264] != v[494] != v[1059], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[148] != v[446] != v[502] != v[1060], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[149] != v[447] != v[503] != v[1061], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[150] != v[448] != v[504] != v[1062], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[151] != v[449] != v[505] != v[1063], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[152] != v[450] != v[506] != v[1064], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[153] != v[451] != v[507] != v[1065], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[154] != v[452] != v[508] != v[1066], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[155] != v[453] != v[509] != v[1067], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[156] != v[454] != v[510] != v[1068], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[157] != v[455] != v[511] != v[1069], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[158] != v[456] != v[512] != v[1070], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[159] != v[457] != v[513] != v[1071], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[140] != v[458] != v[514] != v[1072], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[141] != v[459] != v[515] != v[1073], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[142] != v[440] != v[516] != v[1074], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[143] != v[441] != v[517] != v[1075], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[144] != v[442] != v[518] != v[1076], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[145] != v[443] != v[519] != v[1077], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[146] != v[444] != v[500] != v[1078], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[147] != v[445] != v[501] != v[1079], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[242] != v[291] != v[497] != v[1080], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[243] != v[292] != v[498] != v[1081], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[244] != v[293] != v[499] != v[1082], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[245] != v[294] != v[480] != v[1083], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[246] != v[295] != v[481] != v[1084], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[247] != v[296] != v[482] != v[1085], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[248] != v[297] != v[483] != v[1086], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[249] != v[298] != v[484] != v[1087], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[250] != v[299] != v[485] != v[1088], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[251] != v[280] != v[486] != v[1089], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[252] != v[281] != v[487] != v[1090], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[253] != v[282] != v[488] != v[1091], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[254] != v[283] != v[489] != v[1092], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[255] != v[284] != v[490] != v[1093], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[256] != v[285] != v[491] != v[1094], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[257] != v[286] != v[492] != v[1095], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[258] != v[287] != v[493] != v[1096], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[259] != v[288] != v[494] != v[1097], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[240] != v[289] != v[495] != v[1098], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[241] != v[290] != v[496] != v[1099], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[40] != v[220] != v[433] != v[1100], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[41] != v[221] != v[434] != v[1101], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[42] != v[222] != v[435] != v[1102], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[43] != v[223] != v[436] != v[1103], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[44] != v[224] != v[437] != v[1104], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[45] != v[225] != v[438] != v[1105], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[46] != v[226] != v[439] != v[1106], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[47] != v[227] != v[420] != v[1107], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[48] != v[228] != v[421] != v[1108], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[49] != v[229] != v[422] != v[1109], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[50] != v[230] != v[423] != v[1110], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[51] != v[231] != v[424] != v[1111], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[52] != v[232] != v[425] != v[1112], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[53] != v[233] != v[426] != v[1113], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[54] != v[234] != v[427] != v[1114], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[55] != v[235] != v[428] != v[1115], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[56] != v[236] != v[429] != v[1116], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[57] != v[237] != v[430] != v[1117], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[58] != v[238] != v[431] != v[1118], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[59] != v[239] != v[432] != v[1119], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[156] != v[305] != v[344] != v[1120], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[157] != v[306] != v[345] != v[1121], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[158] != v[307] != v[346] != v[1122], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[159] != v[308] != v[347] != v[1123], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[140] != v[309] != v[348] != v[1124], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[141] != v[310] != v[349] != v[1125], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[142] != v[311] != v[350] != v[1126], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[143] != v[312] != v[351] != v[1127], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[144] != v[313] != v[352] != v[1128], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[145] != v[314] != v[353] != v[1129], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[146] != v[315] != v[354] != v[1130], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[147] != v[316] != v[355] != v[1131], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[148] != v[317] != v[356] != v[1132], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[149] != v[318] != v[357] != v[1133], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[150] != v[319] != v[358] != v[1134], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[151] != v[300] != v[359] != v[1135], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[152] != v[301] != v[340] != v[1136], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[153] != v[302] != v[341] != v[1137], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[154] != v[303] != v[342] != v[1138], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[155] != v[304] != v[343] != v[1139], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[138] != v[255] != v[445] != v[1140], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[139] != v[256] != v[446] != v[1141], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[120] != v[257] != v[447] != v[1142], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[121] != v[258] != v[448] != v[1143], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[122] != v[259] != v[449] != v[1144], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[123] != v[240] != v[450] != v[1145], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[124] != v[241] != v[451] != v[1146], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[125] != v[242] != v[452] != v[1147], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[126] != v[243] != v[453] != v[1148], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[127] != v[244] != v[454] != v[1149], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[128] != v[245] != v[455] != v[1150], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[129] != v[246] != v[456] != v[1151], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[130] != v[247] != v[457] != v[1152], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[131] != v[248] != v[458] != v[1153], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[132] != v[249] != v[459] != v[1154], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[133] != v[250] != v[440] != v[1155], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[134] != v[251] != v[441] != v[1156], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[135] != v[252] != v[442] != v[1157], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[136] != v[253] != v[443] != v[1158], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[137] != v[254] != v[444] != v[1159], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[299] != v[318] != v[360] != v[1160], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[280] != v[319] != v[361] != v[1161], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[281] != v[300] != v[362] != v[1162], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[282] != v[301] != v[363] != v[1163], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[283] != v[302] != v[364] != v[1164], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[284] != v[303] != v[365] != v[1165], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[285] != v[304] != v[366] != v[1166], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[286] != v[305] != v[367] != v[1167], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[287] != v[306] != v[368] != v[1168], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[288] != v[307] != v[369] != v[1169], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[289] != v[308] != v[370] != v[1170], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[290] != v[309] != v[371] != v[1171], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[291] != v[310] != v[372] != v[1172], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[292] != v[311] != v[373] != v[1173], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[293] != v[312] != v[374] != v[1174], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[294] != v[313] != v[375] != v[1175], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[295] != v[314] != v[376] != v[1176], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[296] != v[315] != v[377] != v[1177], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[297] != v[316] != v[378] != v[1178], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[298] != v[317] != v[379] != v[1179], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[275] != v[475] != v[1180], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[276] != v[476] != v[1181], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[277] != v[477] != v[1182], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[278] != v[478] != v[1183], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[279] != v[479] != v[1184], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[260] != v[460] != v[1185], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[261] != v[461] != v[1186], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[262] != v[462] != v[1187], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[263] != v[463] != v[1188], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[264] != v[464] != v[1189], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[265] != v[465] != v[1190], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[266] != v[466] != v[1191], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[267] != v[467] != v[1192], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[268] != v[468] != v[1193], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[269] != v[469] != v[1194], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[270] != v[470] != v[1195], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[271] != v[471] != v[1196], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[272] != v[472] != v[1197], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[273] != v[473] != v[1198], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[274] != v[474] != v[1199], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[11] != v[199] != v[204] != v[241] != v[1200], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[12] != v[180] != v[205] != v[242] != v[1201], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[13] != v[181] != v[206] != v[243] != v[1202], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[14] != v[182] != v[207] != v[244] != v[1203], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[15] != v[183] != v[208] != v[245] != v[1204], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[16] != v[184] != v[209] != v[246] != v[1205], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[17] != v[185] != v[210] != v[247] != v[1206], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[18] != v[186] != v[211] != v[248] != v[1207], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[19] != v[187] != v[212] != v[249] != v[1208], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[0] != v[188] != v[213] != v[250] != v[1209], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[1] != v[189] != v[214] != v[251] != v[1210], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[2] != v[190] != v[215] != v[252] != v[1211], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[3] != v[191] != v[216] != v[253] != v[1212], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[4] != v[192] != v[217] != v[254] != v[1213], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[5] != v[193] != v[218] != v[255] != v[1214], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[6] != v[194] != v[219] != v[256] != v[1215], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[7] != v[195] != v[200] != v[257] != v[1216], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[8] != v[196] != v[201] != v[258] != v[1217], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[9] != v[197] != v[202] != v[259] != v[1218], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[10] != v[198] != v[203] != v[240] != v[1219], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[22] != v[77] != v[152] != v[388] != v[1220], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[23] != v[78] != v[153] != v[389] != v[1221], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[24] != v[79] != v[154] != v[390] != v[1222], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[25] != v[60] != v[155] != v[391] != v[1223], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[26] != v[61] != v[156] != v[392] != v[1224], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[27] != v[62] != v[157] != v[393] != v[1225], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[28] != v[63] != v[158] != v[394] != v[1226], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[29] != v[64] != v[159] != v[395] != v[1227], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[30] != v[65] != v[140] != v[396] != v[1228], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[31] != v[66] != v[141] != v[397] != v[1229], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[32] != v[67] != v[142] != v[398] != v[1230], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[33] != v[68] != v[143] != v[399] != v[1231], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[34] != v[69] != v[144] != v[380] != v[1232], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[35] != v[70] != v[145] != v[381] != v[1233], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[36] != v[71] != v[146] != v[382] != v[1234], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[37] != v[72] != v[147] != v[383] != v[1235], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[38] != v[73] != v[148] != v[384] != v[1236], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[39] != v[74] != v[149] != v[385] != v[1237], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[20] != v[75] != v[150] != v[386] != v[1238], Variable.Bernoulli(0));
                //Variable.ConstrainEqual(v[21] != v[76] != v[151] != v[387] != v[1239], Variable.Bernoulli(0));

                //540-440-0.8
                Variable.ConstrainEqual(v[13] != v[35] != v[43] != v[69] != v[100] != v[139] != v[195] != v[202] != v[235] != v[244] != v[273] != v[318] != v[330] != v[373] != v[396] != v[409] != v[432] != v[441] != v[460], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[14] != v[36] != v[44] != v[70] != v[101] != v[120] != v[196] != v[203] != v[236] != v[245] != v[274] != v[319] != v[331] != v[374] != v[397] != v[410] != v[433] != v[442] != v[461], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[15] != v[37] != v[45] != v[71] != v[102] != v[121] != v[197] != v[204] != v[237] != v[246] != v[275] != v[300] != v[332] != v[375] != v[398] != v[411] != v[434] != v[443] != v[462], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[16] != v[38] != v[46] != v[72] != v[103] != v[122] != v[198] != v[205] != v[238] != v[247] != v[276] != v[301] != v[333] != v[376] != v[399] != v[412] != v[435] != v[444] != v[463], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[17] != v[39] != v[47] != v[73] != v[104] != v[123] != v[199] != v[206] != v[239] != v[248] != v[277] != v[302] != v[334] != v[377] != v[380] != v[413] != v[436] != v[445] != v[464], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[18] != v[20] != v[48] != v[74] != v[105] != v[124] != v[180] != v[207] != v[220] != v[249] != v[278] != v[303] != v[335] != v[378] != v[381] != v[414] != v[437] != v[446] != v[465], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[19] != v[21] != v[49] != v[75] != v[106] != v[125] != v[181] != v[208] != v[221] != v[250] != v[279] != v[304] != v[336] != v[379] != v[382] != v[415] != v[438] != v[447] != v[466], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[22] != v[50] != v[76] != v[107] != v[126] != v[182] != v[209] != v[222] != v[251] != v[260] != v[305] != v[337] != v[360] != v[383] != v[416] != v[439] != v[448] != v[467], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[23] != v[51] != v[77] != v[108] != v[127] != v[183] != v[210] != v[223] != v[252] != v[261] != v[306] != v[338] != v[361] != v[384] != v[417] != v[420] != v[449] != v[468], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[24] != v[52] != v[78] != v[109] != v[128] != v[184] != v[211] != v[224] != v[253] != v[262] != v[307] != v[339] != v[362] != v[385] != v[418] != v[421] != v[450] != v[469], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[25] != v[53] != v[79] != v[110] != v[129] != v[185] != v[212] != v[225] != v[254] != v[263] != v[308] != v[320] != v[363] != v[386] != v[419] != v[422] != v[451] != v[470], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[4] != v[26] != v[54] != v[60] != v[111] != v[130] != v[186] != v[213] != v[226] != v[255] != v[264] != v[309] != v[321] != v[364] != v[387] != v[400] != v[423] != v[452] != v[471], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[5] != v[27] != v[55] != v[61] != v[112] != v[131] != v[187] != v[214] != v[227] != v[256] != v[265] != v[310] != v[322] != v[365] != v[388] != v[401] != v[424] != v[453] != v[472], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[6] != v[28] != v[56] != v[62] != v[113] != v[132] != v[188] != v[215] != v[228] != v[257] != v[266] != v[311] != v[323] != v[366] != v[389] != v[402] != v[425] != v[454] != v[473], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[7] != v[29] != v[57] != v[63] != v[114] != v[133] != v[189] != v[216] != v[229] != v[258] != v[267] != v[312] != v[324] != v[367] != v[390] != v[403] != v[426] != v[455] != v[474], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[8] != v[30] != v[58] != v[64] != v[115] != v[134] != v[190] != v[217] != v[230] != v[259] != v[268] != v[313] != v[325] != v[368] != v[391] != v[404] != v[427] != v[456] != v[475], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[9] != v[31] != v[59] != v[65] != v[116] != v[135] != v[191] != v[218] != v[231] != v[240] != v[269] != v[314] != v[326] != v[369] != v[392] != v[405] != v[428] != v[457] != v[476], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[10] != v[32] != v[40] != v[66] != v[117] != v[136] != v[192] != v[219] != v[232] != v[241] != v[270] != v[315] != v[327] != v[370] != v[393] != v[406] != v[429] != v[458] != v[477], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[11] != v[33] != v[41] != v[67] != v[118] != v[137] != v[193] != v[200] != v[233] != v[242] != v[271] != v[316] != v[328] != v[371] != v[394] != v[407] != v[430] != v[459] != v[478], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[12] != v[34] != v[42] != v[68] != v[119] != v[138] != v[194] != v[201] != v[234] != v[243] != v[272] != v[317] != v[329] != v[372] != v[395] != v[408] != v[431] != v[440] != v[479], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[54] != v[67] != v[81] != v[101] != v[153] != v[164] != v[180] != v[229] != v[240] != v[296] != v[306] != v[332] != v[343] != v[380] != v[421] != v[440] != v[460] != v[480], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[4] != v[55] != v[68] != v[82] != v[102] != v[154] != v[165] != v[181] != v[230] != v[241] != v[297] != v[307] != v[333] != v[344] != v[381] != v[422] != v[441] != v[461] != v[481], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[5] != v[56] != v[69] != v[83] != v[103] != v[155] != v[166] != v[182] != v[231] != v[242] != v[298] != v[308] != v[334] != v[345] != v[382] != v[423] != v[442] != v[462] != v[482], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[6] != v[57] != v[70] != v[84] != v[104] != v[156] != v[167] != v[183] != v[232] != v[243] != v[299] != v[309] != v[335] != v[346] != v[383] != v[424] != v[443] != v[463] != v[483], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[7] != v[58] != v[71] != v[85] != v[105] != v[157] != v[168] != v[184] != v[233] != v[244] != v[280] != v[310] != v[336] != v[347] != v[384] != v[425] != v[444] != v[464] != v[484], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[8] != v[59] != v[72] != v[86] != v[106] != v[158] != v[169] != v[185] != v[234] != v[245] != v[281] != v[311] != v[337] != v[348] != v[385] != v[426] != v[445] != v[465] != v[485], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[9] != v[40] != v[73] != v[87] != v[107] != v[159] != v[170] != v[186] != v[235] != v[246] != v[282] != v[312] != v[338] != v[349] != v[386] != v[427] != v[446] != v[466] != v[486], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[10] != v[41] != v[74] != v[88] != v[108] != v[140] != v[171] != v[187] != v[236] != v[247] != v[283] != v[313] != v[339] != v[350] != v[387] != v[428] != v[447] != v[467] != v[487], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[11] != v[42] != v[75] != v[89] != v[109] != v[141] != v[172] != v[188] != v[237] != v[248] != v[284] != v[314] != v[320] != v[351] != v[388] != v[429] != v[448] != v[468] != v[488], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[12] != v[43] != v[76] != v[90] != v[110] != v[142] != v[173] != v[189] != v[238] != v[249] != v[285] != v[315] != v[321] != v[352] != v[389] != v[430] != v[449] != v[469] != v[489], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[13] != v[44] != v[77] != v[91] != v[111] != v[143] != v[174] != v[190] != v[239] != v[250] != v[286] != v[316] != v[322] != v[353] != v[390] != v[431] != v[450] != v[470] != v[490], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[14] != v[45] != v[78] != v[92] != v[112] != v[144] != v[175] != v[191] != v[220] != v[251] != v[287] != v[317] != v[323] != v[354] != v[391] != v[432] != v[451] != v[471] != v[491], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[15] != v[46] != v[79] != v[93] != v[113] != v[145] != v[176] != v[192] != v[221] != v[252] != v[288] != v[318] != v[324] != v[355] != v[392] != v[433] != v[452] != v[472] != v[492], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[16] != v[47] != v[60] != v[94] != v[114] != v[146] != v[177] != v[193] != v[222] != v[253] != v[289] != v[319] != v[325] != v[356] != v[393] != v[434] != v[453] != v[473] != v[493], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[17] != v[48] != v[61] != v[95] != v[115] != v[147] != v[178] != v[194] != v[223] != v[254] != v[290] != v[300] != v[326] != v[357] != v[394] != v[435] != v[454] != v[474] != v[494], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[18] != v[49] != v[62] != v[96] != v[116] != v[148] != v[179] != v[195] != v[224] != v[255] != v[291] != v[301] != v[327] != v[358] != v[395] != v[436] != v[455] != v[475] != v[495], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[19] != v[50] != v[63] != v[97] != v[117] != v[149] != v[160] != v[196] != v[225] != v[256] != v[292] != v[302] != v[328] != v[359] != v[396] != v[437] != v[456] != v[476] != v[496], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[51] != v[64] != v[98] != v[118] != v[150] != v[161] != v[197] != v[226] != v[257] != v[293] != v[303] != v[329] != v[340] != v[397] != v[438] != v[457] != v[477] != v[497], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[52] != v[65] != v[99] != v[119] != v[151] != v[162] != v[198] != v[227] != v[258] != v[294] != v[304] != v[330] != v[341] != v[398] != v[439] != v[458] != v[478] != v[498], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[53] != v[66] != v[80] != v[100] != v[152] != v[163] != v[199] != v[228] != v[259] != v[295] != v[305] != v[331] != v[342] != v[399] != v[420] != v[459] != v[479] != v[499], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[8] != v[27] != v[40] != v[80] != v[118] != v[127] != v[142] != v[160] != v[191] != v[206] != v[275] != v[283] != v[301] != v[344] != v[366] != v[390] != v[416] != v[480] != v[500], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[9] != v[28] != v[41] != v[81] != v[119] != v[128] != v[143] != v[161] != v[192] != v[207] != v[276] != v[284] != v[302] != v[345] != v[367] != v[391] != v[417] != v[481] != v[501], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[10] != v[29] != v[42] != v[82] != v[100] != v[129] != v[144] != v[162] != v[193] != v[208] != v[277] != v[285] != v[303] != v[346] != v[368] != v[392] != v[418] != v[482] != v[502], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[11] != v[30] != v[43] != v[83] != v[101] != v[130] != v[145] != v[163] != v[194] != v[209] != v[278] != v[286] != v[304] != v[347] != v[369] != v[393] != v[419] != v[483] != v[503], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[12] != v[31] != v[44] != v[84] != v[102] != v[131] != v[146] != v[164] != v[195] != v[210] != v[279] != v[287] != v[305] != v[348] != v[370] != v[394] != v[400] != v[484] != v[504], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[13] != v[32] != v[45] != v[85] != v[103] != v[132] != v[147] != v[165] != v[196] != v[211] != v[260] != v[288] != v[306] != v[349] != v[371] != v[395] != v[401] != v[485] != v[505], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[14] != v[33] != v[46] != v[86] != v[104] != v[133] != v[148] != v[166] != v[197] != v[212] != v[261] != v[289] != v[307] != v[350] != v[372] != v[396] != v[402] != v[486] != v[506], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[15] != v[34] != v[47] != v[87] != v[105] != v[134] != v[149] != v[167] != v[198] != v[213] != v[262] != v[290] != v[308] != v[351] != v[373] != v[397] != v[403] != v[487] != v[507], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[16] != v[35] != v[48] != v[88] != v[106] != v[135] != v[150] != v[168] != v[199] != v[214] != v[263] != v[291] != v[309] != v[352] != v[374] != v[398] != v[404] != v[488] != v[508], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[17] != v[36] != v[49] != v[89] != v[107] != v[136] != v[151] != v[169] != v[180] != v[215] != v[264] != v[292] != v[310] != v[353] != v[375] != v[399] != v[405] != v[489] != v[509], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[18] != v[37] != v[50] != v[90] != v[108] != v[137] != v[152] != v[170] != v[181] != v[216] != v[265] != v[293] != v[311] != v[354] != v[376] != v[380] != v[406] != v[490] != v[510], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[19] != v[38] != v[51] != v[91] != v[109] != v[138] != v[153] != v[171] != v[182] != v[217] != v[266] != v[294] != v[312] != v[355] != v[377] != v[381] != v[407] != v[491] != v[511], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[39] != v[52] != v[92] != v[110] != v[139] != v[154] != v[172] != v[183] != v[218] != v[267] != v[295] != v[313] != v[356] != v[378] != v[382] != v[408] != v[492] != v[512], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[20] != v[53] != v[93] != v[111] != v[120] != v[155] != v[173] != v[184] != v[219] != v[268] != v[296] != v[314] != v[357] != v[379] != v[383] != v[409] != v[493] != v[513], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[21] != v[54] != v[94] != v[112] != v[121] != v[156] != v[174] != v[185] != v[200] != v[269] != v[297] != v[315] != v[358] != v[360] != v[384] != v[410] != v[494] != v[514], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[22] != v[55] != v[95] != v[113] != v[122] != v[157] != v[175] != v[186] != v[201] != v[270] != v[298] != v[316] != v[359] != v[361] != v[385] != v[411] != v[495] != v[515], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[4] != v[23] != v[56] != v[96] != v[114] != v[123] != v[158] != v[176] != v[187] != v[202] != v[271] != v[299] != v[317] != v[340] != v[362] != v[386] != v[412] != v[496] != v[516], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[5] != v[24] != v[57] != v[97] != v[115] != v[124] != v[159] != v[177] != v[188] != v[203] != v[272] != v[280] != v[318] != v[341] != v[363] != v[387] != v[413] != v[497] != v[517], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[6] != v[25] != v[58] != v[98] != v[116] != v[125] != v[140] != v[178] != v[189] != v[204] != v[273] != v[281] != v[319] != v[342] != v[364] != v[388] != v[414] != v[498] != v[518], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[7] != v[26] != v[59] != v[99] != v[117] != v[126] != v[141] != v[179] != v[190] != v[205] != v[274] != v[282] != v[300] != v[343] != v[365] != v[389] != v[415] != v[499] != v[519], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[28] != v[70] != v[97] != v[121] != v[155] != v[179] != v[201] != v[223] != v[253] != v[264] != v[293] != v[324] != v[348] != v[377] != v[419] != v[428] != v[441] != v[500], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[29] != v[71] != v[98] != v[122] != v[156] != v[160] != v[202] != v[224] != v[254] != v[265] != v[294] != v[325] != v[349] != v[378] != v[400] != v[429] != v[442] != v[501], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[30] != v[72] != v[99] != v[123] != v[157] != v[161] != v[203] != v[225] != v[255] != v[266] != v[295] != v[326] != v[350] != v[379] != v[401] != v[430] != v[443] != v[502], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[31] != v[73] != v[80] != v[124] != v[158] != v[162] != v[204] != v[226] != v[256] != v[267] != v[296] != v[327] != v[351] != v[360] != v[402] != v[431] != v[444] != v[503], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[4] != v[32] != v[74] != v[81] != v[125] != v[159] != v[163] != v[205] != v[227] != v[257] != v[268] != v[297] != v[328] != v[352] != v[361] != v[403] != v[432] != v[445] != v[504], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[5] != v[33] != v[75] != v[82] != v[126] != v[140] != v[164] != v[206] != v[228] != v[258] != v[269] != v[298] != v[329] != v[353] != v[362] != v[404] != v[433] != v[446] != v[505], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[6] != v[34] != v[76] != v[83] != v[127] != v[141] != v[165] != v[207] != v[229] != v[259] != v[270] != v[299] != v[330] != v[354] != v[363] != v[405] != v[434] != v[447] != v[506], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[7] != v[35] != v[77] != v[84] != v[128] != v[142] != v[166] != v[208] != v[230] != v[240] != v[271] != v[280] != v[331] != v[355] != v[364] != v[406] != v[435] != v[448] != v[507], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[8] != v[36] != v[78] != v[85] != v[129] != v[143] != v[167] != v[209] != v[231] != v[241] != v[272] != v[281] != v[332] != v[356] != v[365] != v[407] != v[436] != v[449] != v[508], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[9] != v[37] != v[79] != v[86] != v[130] != v[144] != v[168] != v[210] != v[232] != v[242] != v[273] != v[282] != v[333] != v[357] != v[366] != v[408] != v[437] != v[450] != v[509], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[10] != v[38] != v[60] != v[87] != v[131] != v[145] != v[169] != v[211] != v[233] != v[243] != v[274] != v[283] != v[334] != v[358] != v[367] != v[409] != v[438] != v[451] != v[510], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[11] != v[39] != v[61] != v[88] != v[132] != v[146] != v[170] != v[212] != v[234] != v[244] != v[275] != v[284] != v[335] != v[359] != v[368] != v[410] != v[439] != v[452] != v[511], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[12] != v[20] != v[62] != v[89] != v[133] != v[147] != v[171] != v[213] != v[235] != v[245] != v[276] != v[285] != v[336] != v[340] != v[369] != v[411] != v[420] != v[453] != v[512], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[13] != v[21] != v[63] != v[90] != v[134] != v[148] != v[172] != v[214] != v[236] != v[246] != v[277] != v[286] != v[337] != v[341] != v[370] != v[412] != v[421] != v[454] != v[513], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[14] != v[22] != v[64] != v[91] != v[135] != v[149] != v[173] != v[215] != v[237] != v[247] != v[278] != v[287] != v[338] != v[342] != v[371] != v[413] != v[422] != v[455] != v[514], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[15] != v[23] != v[65] != v[92] != v[136] != v[150] != v[174] != v[216] != v[238] != v[248] != v[279] != v[288] != v[339] != v[343] != v[372] != v[414] != v[423] != v[456] != v[515], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[16] != v[24] != v[66] != v[93] != v[137] != v[151] != v[175] != v[217] != v[239] != v[249] != v[260] != v[289] != v[320] != v[344] != v[373] != v[415] != v[424] != v[457] != v[516], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[17] != v[25] != v[67] != v[94] != v[138] != v[152] != v[176] != v[218] != v[220] != v[250] != v[261] != v[290] != v[321] != v[345] != v[374] != v[416] != v[425] != v[458] != v[517], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[18] != v[26] != v[68] != v[95] != v[139] != v[153] != v[177] != v[219] != v[221] != v[251] != v[262] != v[291] != v[322] != v[346] != v[375] != v[417] != v[426] != v[459] != v[518], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[19] != v[27] != v[69] != v[96] != v[120] != v[154] != v[178] != v[200] != v[222] != v[252] != v[263] != v[292] != v[323] != v[347] != v[376] != v[418] != v[427] != v[440] != v[519], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[13] != v[25] != v[520], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[14] != v[26] != v[521], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[15] != v[27] != v[522], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[16] != v[28] != v[523], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[17] != v[29] != v[524], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[18] != v[30] != v[525], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[19] != v[31] != v[526], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[32] != v[527], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[33] != v[528], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[34] != v[529], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[35] != v[530], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[4] != v[36] != v[531], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[5] != v[37] != v[532], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[6] != v[38] != v[533], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[7] != v[39] != v[534], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[8] != v[20] != v[535], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[9] != v[21] != v[536], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[10] != v[22] != v[537], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[11] != v[23] != v[538], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[12] != v[24] != v[539], Variable.Bernoulli(0));


                if (InferenceEngine == null)
                {
                    InferenceEngine = new InferenceEngine(new ExpectationPropagation());
                    InferenceEngine.ShowProgress = false;
                    InferenceEngine.NumberOfIterations = 100;

                }
            }

            public virtual void SetModelData(ModelData priors)
            {
                vPriors.ObservedValue = priors.vBernoulliDistArray;
            }
        }

        public struct ModelData
        {
            public Bernoulli[] vBernoulliDistArray;

            public ModelData(double[] zNodeArray, int n)
            {
                vBernoulliDistArray = new Bernoulli[n];

                for (int i = 0; i < n; i++)
                {
                    vBernoulliDistArray[i] = new Bernoulli(zNodeArray[i]);
                }

            }
        }

        public class CyclistTraining : CyclistBase
        {
            public override void CreateModel()
            {
                base.CreateModel();
                // not sure what I need here
            }

            public ModelData InferModelData()
            {
                ModelData posteriors;
                posteriors.vBernoulliDistArray = InferenceEngine.Infer<Bernoulli[]>(v);
                return posteriors;
            }
        }

        static void Main(string[] args)
        {
            // Here we define the dimensions of the codeword
            int n = 540;  // number of variable nodes  (88, 124, 54) (880, 1240, 540)
            //int k = 44;   // number of message bits  (44, 44, 44) (440, 440, 440)
            int packets = 23188; //(142290, 100980, 231880) (14229, 10098, 23188)

            // Here we define where the data comes from
            double[,] data = new double[packets, n];

            String line = String.Empty;

            //System.IO.StreamReader file = new System.IO.StreamReader("NR_1_0_2_packets_231880_rate_0.8_n_54_m_10.csv");
            //System.IO.StreamReader file = new System.IO.StreamReader("NR_1_0_2_packets_142290_rate_0.5_n_88_m_44.csv");
            //System.IO.StreamReader file = new System.IO.StreamReader("NR_1_0_2_packets_100980_rate_0.35_n_124_m_80.csv");
            //System.IO.StreamReader file = new System.IO.StreamReader("NR_1_2_20_packets_10098_rate_0.35_n_1240_m_800.csv");
            //System.IO.StreamReader file = new System.IO.StreamReader("NR_1_2_20_packets_14229_rate_0.5_n_880_m_440.csv");
            System.IO.StreamReader file = new System.IO.StreamReader("NR_1_2_20_packets_23188_rate_0.8_n_540_m_100.csv");


            int iter = 0;
            while ((line = file.ReadLine()) != null)
            {
                String[] parts_of_line = line.Split(',');
                for (int i = 0; i < parts_of_line.Length; i++)
                {
                    parts_of_line[i] = parts_of_line[i].Trim();
                    data[iter, i] = double.Parse(parts_of_line[i], System.Globalization.CultureInfo.InvariantCulture);
                }
                iter++;
            }

            Console.WriteLine("\nCodeword data imported...");

            double[] precisions = new double[12521520];
            file = new System.IO.StreamReader("precision-12521520.csv");

            iter = 0;
            while ((line = file.ReadLine()) != null)
            {
                String[] parts_of_line = line.Split(',');
                for (int i = 0; i < parts_of_line.Length; i++)
                {
                    parts_of_line[i] = parts_of_line[i].Trim();
                    precisions[iter] = double.Parse(parts_of_line[i], System.Globalization.CultureInfo.InvariantCulture);
                }
                iter++;
            }

            Console.WriteLine("Precision data imported...");

            // Here we create the model
            CyclistTraining cyclistTraining = new CyclistTraining();
            cyclistTraining.CreateModel();

            Console.WriteLine("Model created...");

            //--------------------------Simulation------------------------------

            Rand.Restart(12347);

            double[,] ab1 = new double[n, 2];
            double[] prevNat1 = new double[] { 0.0, 0.0 };
            int bitCounter1 = 0;
            var tuple1 = new Tuple<double[,], double[], int>(ab1, prevNat1, bitCounter1);

            double[,] ab2 = new double[n, 2];
            double[] prevNat2 = new double[] { 0.0, 0.0 };
            int bitCounter2 = 0;
            var tuple2 = new Tuple<double[,], double[], int>(ab2, prevNat2, bitCounter2);

            Console.WriteLine("Run simulation...");

            for (int N = 0; N < packets; N++)
            {
                double[] tbits = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (data[N, i] == 0)
                    {
                        tbits[i] = -1;
                    }
                    else
                    {
                        tbits[i] = 1;
                    }
                }

                double[] rbits = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double true_precision = precisions[(N * n) + i];
                    rbits[i] = tbits[i] + Rand.Normal(0, Math.Sqrt(1 / true_precision));
                }

                double[] zValues = new double[n];
                double[] zValues1 = new double[n];
                double[] zValues2 = new double[n];
                double[] Gammas = new double[n];
                double[] Gammas1 = new double[n];
                double[] Gammas_var1 = new double[n];
                double[] Gammas2 = new double[n];
                double[] Gammas_var2 = new double[n];

                //// Here we initialise the z values
                for (int i = 0; i < n; i++)
                {
                    zValues[i] = 0.5;
                }

                ModelData initPriors = new ModelData();
                ModelData zPosteriors = new ModelData();

                // here we need to get the best code performance from known Gamma
                for (int i = 0; i < n; i++)
                {
                    Gammas[i] = precisions[(N * n) + i];
                }

                for (int i = 0; i < 1; i++)
                {
                    zValues = ZResponsibilities(Gammas, rbits);

                    initPriors = new ModelData(zValues, zValues.Length);
                    cyclistTraining.SetModelData(initPriors);
                    zPosteriors = cyclistTraining.InferModelData();

                    for (int j = 0; j < n; j++)
                    {
                        zValues[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                    }
                }

                //// Here we initialise the z values for model 1
                for (int i = 0; i < n; i++)
                {
                    zValues1[i] = 0.5;
                }

                // here we need to alternate between models
                for (int i = 0; i < 3; i++)
                {
                    tuple1 = NoiseEstimation(15, 1.0, 0.1, prevNat1, rbits, zValues1, bitCounter1);

                    for (int j = 0; j < n; j++)
                    {
                        Gammas1[j] = tuple1.Item1[j, 0] / tuple1.Item1[j, 1];
                        Gammas_var1[j] = tuple1.Item1[j, 0] / Math.Pow(tuple1.Item1[j, 1], 2);
                    }

                    zValues1 = ZResponsibilities(Gammas1, rbits);

                    initPriors = new ModelData(zValues1, zValues1.Length);
                    cyclistTraining.SetModelData(initPriors);
                    zPosteriors = cyclistTraining.InferModelData();

                    for (int j = 0; j < n; j++)
                    {
                        zValues1[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                    }

                }

                prevNat1 = tuple1.Item2;
                bitCounter1 = n * (N + 1);
                //// -------------------------------------- ////
                
                //// Here we initialise the z values for model 2
                for (int i = 0; i < n; i++)
                {
                    zValues2[i] = 0.5;
                }

                // here we need to alternate between models
                for (int i = 0; i < 3; i++)
                {
                    tuple2 = NoiseEstimation(450, 1.0, 1.0, prevNat2, rbits, zValues2, bitCounter2);

                    for (int j = 0; j < n; j++)
                    {
                        Gammas2[j] = tuple2.Item1[j, 0] / tuple2.Item1[j, 1];
                        Gammas_var2[j] = tuple2.Item1[j, 0] / Math.Pow(tuple2.Item1[j, 1], 2);
                    }

                    zValues2 = ZResponsibilities(Gammas2, rbits);

                    initPriors = new ModelData(zValues2, zValues2.Length);
                    cyclistTraining.SetModelData(initPriors);
                    zPosteriors = cyclistTraining.InferModelData();

                    for (int j = 0; j < n; j++)
                    {
                        zValues2[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                    }

                }

                prevNat2 = tuple2.Item2;
                bitCounter2 = n * (N + 1);
                
                var storeEstPrec = new StringBuilder();

                for (int i = 0; i < n; i++)
                {
                    var newLine = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}",
                        tbits[i],
                        precisions[(N * n) + i],
                        rbits[i],
                        zValues1[i],
                        Gammas1[i],
                        Gammas_var1[i],
                        zValues2[i],
                        Gammas2[i],
                        Gammas_var2[i],
                        zValues[i]);
                    storeEstPrec.AppendLine(newLine);
                }

                File.AppendAllText("bit-level-noise-estimation-540-440.csv", storeEstPrec.ToString());

            }

            Console.WriteLine("Bits counter 1 (12521520): ", bitCounter1);
            Console.WriteLine("Bits counter 2 (12521520): ", bitCounter2);
            Console.WriteLine("--------------DONE---------------");

        }

    }

}