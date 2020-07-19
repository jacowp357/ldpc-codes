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
        static Tuple<double[], double[], int> NoiseEstimation(double Nmax, double aPrior, double bPrior, double[] prevGammaNatural, double[] r, double[] z, int bitCount)
        {
            double mean0 = -1.0;
            double mean1 = 1.0;

            //double[,] ab = new double[r.Length, 2];
            double[] ab = new double[2];

            //double Nmax = 20;

            double[] GammaNaturalPrior = { -bPrior, aPrior - 1.0 };
            double[] GammaNatural = { prevGammaNatural[0], prevGammaNatural[1] };
            double[] GammaNaturalPosterior = { 0.0, 0.0 };

            for (int i = 0; i < r.Length; i++)
            {
                double zTrue = z[i];
                double zFalse = 1.0 - zTrue;

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

                //GammaNaturalPosterior[0] = GammaNaturalPrior[0] + GammaNatural[0];
                //GammaNaturalPosterior[1] = GammaNaturalPrior[1] + GammaNatural[1];

                //ab[i, 0] = GammaNaturalPosterior[1] + 1.0;      // a parameter
                //ab[i, 1] = -1.0 * GammaNaturalPosterior[0];     // b parameter

                bitCount++;
            }

            GammaNaturalPosterior[0] = GammaNatural[0] + GammaNaturalPrior[0];
            GammaNaturalPosterior[1] = GammaNatural[1] + GammaNaturalPrior[1];

            ab[0] = GammaNaturalPosterior[1] + 1.0;      // a parameter
            ab[1] = -1.0 * GammaNaturalPosterior[0];     // b parameter

            return new Tuple<double[], double[], int>(ab, GammaNatural, bitCount);
        }
        // Here we specify the model by using a baseclass
        public class CyclistBase
        {
            public InferenceEngine InferenceEngine;

            protected VariableArray<bool> v;
            protected VariableArray<Bernoulli> vPriors;

            public virtual void CreateModel()
            {
                Range n = new Range(88);

                v = Variable.Array<bool>(n).Named("v");

                vPriors = Variable.Array<Bernoulli>(n);

                v[n] = Variable.Random<bool, Bernoulli>(vPriors[n]);

                //88-44-0.5
                Variable.ConstrainEqual(v[0] != v[3] != v[4] != v[7] != v[10] != v[12] != v[19] != v[21] != v[22] != v[25] != v[27] != v[31] != v[33] != v[36] != v[39] != v[41] != v[43] != v[45] != v[46], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[5] != v[6] != v[11] != v[13] != v[18] != v[20] != v[23] != v[24] != v[26] != v[30] != v[32] != v[37] != v[38] != v[40] != v[42] != v[44] != v[47], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[5] != v[7] != v[8] != v[11] != v[14] != v[16] != v[19] != v[22] != v[24] != v[29] != v[30] != v[32] != v[35] != v[39] != v[42] != v[44] != v[46] != v[48], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[4] != v[6] != v[9] != v[10] != v[15] != v[17] != v[18] != v[23] != v[25] != v[28] != v[31] != v[33] != v[34] != v[38] != v[43] != v[45] != v[47] != v[49], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[3] != v[5] != v[9] != v[11] != v[13] != v[15] != v[17] != v[19] != v[21] != v[26] != v[29] != v[31] != v[35] != v[37] != v[39] != v[41] != v[48] != v[50], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[4] != v[8] != v[10] != v[12] != v[14] != v[16] != v[18] != v[20] != v[27] != v[28] != v[30] != v[34] != v[36] != v[38] != v[40] != v[49] != v[51], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[8] != v[12] != v[15] != v[17] != v[20] != v[22] != v[24] != v[27] != v[29] != v[32] != v[34] != v[36] != v[40] != v[42] != v[45] != v[50], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[9] != v[13] != v[14] != v[16] != v[21] != v[23] != v[25] != v[26] != v[28] != v[33] != v[35] != v[37] != v[41] != v[43] != v[44] != v[51], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[52], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[3] != v[53], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[6] != v[25] != v[32] != v[43] != v[45] != v[54], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[3] != v[7] != v[24] != v[33] != v[42] != v[44] != v[55], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[12] != v[20] != v[23] != v[26] != v[35] != v[37] != v[41] != v[56], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[13] != v[21] != v[22] != v[27] != v[34] != v[36] != v[40] != v[57], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[9] != v[15] != v[17] != v[28] != v[58], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[8] != v[14] != v[16] != v[29] != v[59], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[7] != v[25] != v[32] != v[38] != v[43] != v[45] != v[48] != v[60], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[6] != v[24] != v[33] != v[39] != v[42] != v[44] != v[49] != v[61], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[21] != v[23] != v[26] != v[34] != v[37] != v[40] != v[62], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[3] != v[20] != v[22] != v[27] != v[35] != v[36] != v[41] != v[63], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[5] != v[9] != v[14] != v[17] != v[28] != v[64], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[4] != v[8] != v[15] != v[16] != v[29] != v[65], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[25] != v[32] != v[42] != v[44] != v[46] != v[66], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[24] != v[33] != v[43] != v[45] != v[47] != v[67], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[22] != v[26] != v[37] != v[68], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[23] != v[27] != v[36] != v[69], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[6] != v[15] != v[41] != v[46] != v[70], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[7] != v[14] != v[40] != v[47] != v[71], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[25] != v[30] != v[33] != v[34] != v[43] != v[72], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[24] != v[31] != v[32] != v[35] != v[42] != v[73], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[2] != v[21] != v[27] != v[37] != v[51] != v[74], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[3] != v[20] != v[26] != v[36] != v[50] != v[75], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[7] != v[23] != v[41] != v[44] != v[76], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[6] != v[22] != v[40] != v[45] != v[77], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[28] != v[33] != v[35] != v[42] != v[78], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[29] != v[32] != v[34] != v[43] != v[79], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[25] != v[26] != v[37] != v[39] != v[80], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[24] != v[27] != v[36] != v[38] != v[81], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[3] != v[14] != v[17] != v[20] != v[82], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[2] != v[15] != v[16] != v[21] != v[83], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[1] != v[6] != v[19] != v[23] != v[44] != v[84], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[7] != v[18] != v[22] != v[45] != v[85], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[3] != v[11] != v[33] != v[41] != v[43] != v[86], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[10] != v[32] != v[40] != v[42] != v[87], Variable.Bernoulli(0)); 

                if (InferenceEngine == null)
                {
                    InferenceEngine = new InferenceEngine(new ExpectationPropagation());
                    InferenceEngine.ShowProgress = false;
                    InferenceEngine.NumberOfIterations = 50;
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
            int n = 88;
            //int k = 44;
            int packets = 120000;

            // Here we define where the data comes from
            double[,] data = new double[packets, n];

            String line = String.Empty;

            System.IO.StreamReader file = new System.IO.StreamReader("NR_1_0_2_packets_120000_rate_0.5_n_88_m_44.csv");

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

            double[,] precisions = new double[packets, 2];
            file = new System.IO.StreamReader("precision-120000.csv");

            iter = 0;
            while ((line = file.ReadLine()) != null)
            {
                String[] parts_of_line = line.Split(',');
                for (int i = 0; i < parts_of_line.Length; i++)
                {
                    parts_of_line[i] = parts_of_line[i].Trim();
                    precisions[iter, i] = double.Parse(parts_of_line[i], System.Globalization.CultureInfo.InvariantCulture);
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

            double[] ab1 = new double[2];
            double[] prevNat1 = new double[] { 0.0, 0.0 };
            int bitCounter1 = 0;
            var tuple1 = new Tuple<double[], double[], int>(ab1, prevNat1, bitCounter1);

            double[] ab2 = new double[2];
            double[] prevNat2 = new double[] { 0.0, 0.0 };
            int bitCounter2 = 0;
            var tuple2 = new Tuple<double[], double[], int>(ab2, prevNat2, bitCounter2);

            double alpha1 = 0;
            double alpha2 = 0;
            double beta1 = 0;
            double beta2 = 0;

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

                double sample_precision = precisions[N, 0];

                double[] rbits = new double[n];
                for (int i = 0; i < n; i++)
                {   
                    rbits[i] = tbits[i] + Rand.Normal(0, Math.Sqrt(1 / sample_precision));
                }

                double[] zValues = new double[n];
                double[] zValues1 = new double[n];
                double[] zValues2 = new double[n];
                double[] Gammas = new double[n];
                double[] Gammas1 = new double[n];
                double[] Gammas_var1 = new double[n];
                double[] Gammas2 = new double[n];
                double[] Gammas_var2 = new double[n];

                // Here we initialise the z values
                for (int i = 0; i < n; i++)
                {
                    zValues[i] = 0.5;
                }

                ModelData initPriors = new ModelData();
                ModelData zPosteriors = new ModelData();

                // here we need to get the best code performance from known Gamma
                for (int i = 0; i < n; i++)
                {
                    Gammas[i] = sample_precision;
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
                for (int i = 0; i < 5; i++)
                {
                    tuple1 = NoiseEstimation(500, 3.2, 1.0, prevNat1, rbits, zValues1, bitCounter1);

                    alpha1 = tuple1.Item1[0];
                    beta1 = tuple1.Item1[1];

                    for (int j = 0; j < n; j++)
                    {
                        Gammas1[j] = alpha1 / beta1;
                        Gammas_var1[j] = alpha1 / Math.Pow(beta1, 2);
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

                // -------------------------------------- //

                // Here we initialise the z values for model 2
                for (int i = 0; i < n; i++)
                {
                    zValues2[i] = 0.5;
                }

                // here we need to alternate between models
                for (int i = 0; i < 5; i++)
                {
                    tuple2 = NoiseEstimation(50, 3.2, 1.0, prevNat2, rbits, zValues2, bitCounter2);

                    alpha2 = tuple2.Item1[0];
                    beta2 = tuple2.Item1[1];

                    for (int j = 0; j < n; j++)
                    {
                        Gammas2[j] = alpha2 / beta2;
                        Gammas_var2[j] = alpha2 / Math.Pow(beta2, 2);
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
                        precisions[N, 1],
                        rbits[i],
                        zValues1[i],
                        alpha1,
                        beta1,
                        zValues2[i],
                        alpha2,
                        beta2,
                        zValues[i]);
                    storeEstPrec.AppendLine(newLine);
                }

                File.AppendAllText("packet-level-noise-estimation-88-44-static.csv", storeEstPrec.ToString());

            }

            Console.WriteLine("Bits counter 1: ", bitCounter1);
            Console.WriteLine("Bits counter 2: ", bitCounter2);
            Console.WriteLine("--------------DONE---------------");

        }

    }

}