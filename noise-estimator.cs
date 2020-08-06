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
        static Tuple<double[], double[], int> NoiseEstimation(double Nmax, double aPrior, double bPrior, double[] prevGammaNatural, double[] r, double[] z, int packetCount)
        {
            double mean0 = -1.0;
            double mean1 = 1.0;

            double[] ab = new double[2];

            double[] GammaNaturalPrior = { -bPrior, aPrior - 1.0 };
            double[] GammaNatural = { 0, 0 };
            double[] GammaNaturalPosterior = { 0.0, 0.0 };

            for (int i = 0; i < r.Length; i++)
            {
                double zTrue = z[i];
                double zFalse = 1.0 - zTrue;

                GammaNatural[0] += ((zTrue * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean1) + Math.Pow(mean1, 2))) +
                                        (zFalse * -0.5 * (Math.Pow(r[i], 2) - (2.0 * r[i] * mean0) + Math.Pow(mean0, 2))));
                GammaNatural[1] += ((zTrue * 0.5) + (zFalse * 0.5));

            }

            if (packetCount > Nmax)
            {
                GammaNatural[0] = (prevGammaNatural[0] + GammaNatural[0]) * (Nmax / (Nmax + 1));
                GammaNatural[1] = (prevGammaNatural[1] + GammaNatural[1]) * (Nmax / (Nmax + 1));
            }
            else
            {
                GammaNatural[0] += prevGammaNatural[0];
                GammaNatural[1] += prevGammaNatural[1];
            }

            GammaNaturalPosterior[0] = GammaNatural[0] + GammaNaturalPrior[0];
            GammaNaturalPosterior[1] = GammaNatural[1] + GammaNaturalPrior[1];

            ab[0] = GammaNaturalPosterior[1] + 1.0;      // a parameter
            ab[1] = -1.0 * GammaNaturalPosterior[0];     // b parameter

            packetCount++;

            return new Tuple<double[], double[], int>(ab, GammaNatural, packetCount);
        }
        // Here we specify the model by using a baseclass
        public class CyclistBase
        {
            public InferenceEngine InferenceEngine;

            protected VariableArray<bool> v;
            protected VariableArray<Bernoulli> vPriors;

            public virtual void CreateModel()
            {
                Range n = new Range(8);

                v = Variable.Array<bool>(n).Named("v");

                vPriors = Variable.Array<Bernoulli>(n);

                v[n] = Variable.Random<bool, Bernoulli>(vPriors[n]);

                // LDPC (8,4) code:
                // H = [[1, 1, 0, 0, 1, 0, 1, 0],
                //      [0, 0, 1, 0, 0, 1, 1, 1],
                //      [0, 0, 1, 1, 1, 0, 0, 1],
                //      [1, 1, 0, 1, 0, 1, 0, 0]]

                //these factors should enforce even parity among the connected bits
                Variable.ConstrainEqual(v[0] != v[1] != v[4] != v[6], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[5] != v[6] != v[7], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[2] != v[3] != v[4] != v[7], Variable.Bernoulli(0));
                Variable.ConstrainEqual(v[0] != v[1] != v[3] != v[5], Variable.Bernoulli(0));

                if (InferenceEngine == null)
                {
                    InferenceEngine = new InferenceEngine(new ExpectationPropagation());
                    InferenceEngine.ShowProgress = false;
                    InferenceEngine.NumberOfIterations = 1;
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
            int n = 8;
            int packets = 2000;

            // here we define where the data comes from
            double[,] data = new double[10000, n];

            String line = String.Empty;

            System.IO.StreamReader file = new System.IO.StreamReader("8-5-LDPC.csv");

            int iter = 0;
            while ((line = file.ReadLine()) != null)
            {
                String[] parts_of_line = line.Split(';');
                for (int i = 0; i < parts_of_line.Length; i++)
                {
                    parts_of_line[i] = parts_of_line[i].Trim();
                    data[iter, i] = double.Parse(parts_of_line[i], System.Globalization.CultureInfo.InvariantCulture);
                }
                iter++;
            }

            Console.WriteLine("\nCodeword data imported...");

            double[,] precisions = new double[16000, 1];
            file = new System.IO.StreamReader("precision-8-5-ldpc.csv");

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

            Rand.Restart(12347);

            double[] fastPrevNat = new double[] { 0.0, 0.0 };
            double[] fastAB = new double[2];
            var fastNoiseTuple = new Tuple<double[], double[], int>(fastAB, fastPrevNat, 1);

            double[] slowPrevNat = new double[] { 0.0, 0.0 };
            double[] slowAB = new double[2];
            var slowNoiseTuple = new Tuple<double[], double[], int>(slowAB, slowPrevNat, 1);

            int packetCounter = 0;
            int bitCounter = 0;

            CyclistTraining cyclistTraining = new CyclistTraining();
            cyclistTraining.CreateModel();
            cyclistTraining.InferenceEngine.NumberOfIterations = 1;

            CyclistTraining idealmodel = new CyclistTraining();
            idealmodel.CreateModel();
            idealmodel.InferenceEngine.NumberOfIterations = 50;

            Console.WriteLine("Model created...");

            //--------------------------Simulation----------------------------//
            for (int N = 0; N < packets; N++)
            {
                bitCounter = (n * N);

                double[] shiftedNoisyBits = new double[n];

                for (int i = 0; i < n; i++)
                {
                    double sample_precision = precisions[bitCounter + i, 0];

                    if (data[N, i] == 0)
                    {
                        shiftedNoisyBits[i] = -1 + Rand.Normal(0, Math.Sqrt(1 / sample_precision));
                    }
                    else
                    {
                        shiftedNoisyBits[i] = 1 + Rand.Normal(0, Math.Sqrt(1 / sample_precision));
                    }
                }

                //------fast model------
                double[] zfastValues = new double[n];

                for (int i = 0; i < n; i++)
                {
                    zfastValues[i] = 0.5;
                }

                double[] fastGammas = new double[n];
                double fastAlpha = 0;
                double fastBeta = 0;

                ModelData initPriors = new ModelData();
                ModelData zPosteriors = new ModelData();

                // message-passing over the graph
                for (int i = 0; i < 15; i++)
                {
                    fastNoiseTuple = NoiseEstimation(5, 0.3, 0.1, fastPrevNat, shiftedNoisyBits, zfastValues, packetCounter);

                    fastAlpha = fastNoiseTuple.Item1[0];
                    fastBeta = fastNoiseTuple.Item1[1];

                    for (int j = 0; j < n; j++)
                    {
                        fastGammas[j] = fastAlpha / fastBeta;
                    }

                    zfastValues = ZResponsibilities(fastGammas, shiftedNoisyBits);

                    initPriors = new ModelData(zfastValues, zfastValues.Length);
                    cyclistTraining.SetModelData(initPriors);
                    zPosteriors = cyclistTraining.InferModelData();

                    for (int j = 0; j < n; j++)
                    {
                        zfastValues[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                    }
                }

                fastPrevNat = fastNoiseTuple.Item2;
                //------fast model end------

                //------slow model------
                double[] zslowValues = new double[n];

                for (int i = 0; i < n; i++)
                {
                    zslowValues[i] = 0.5;
                }

                double[] slowGammas = new double[n];
                double slowAlpha = 0;
                double slowBeta = 0;

                // message-passing over the graph
                for (int i = 0; i < 15; i++)
                {
                    slowNoiseTuple = NoiseEstimation(10000000, 3, 1, slowPrevNat, shiftedNoisyBits, zslowValues, packetCounter);

                    slowAlpha = slowNoiseTuple.Item1[0];
                    slowBeta = slowNoiseTuple.Item1[1];

                    for (int j = 0; j < n; j++)
                    {
                        slowGammas[j] = slowAlpha / slowBeta;
                    }

                    zslowValues = ZResponsibilities(slowGammas, shiftedNoisyBits);

                    initPriors = new ModelData(zslowValues, zslowValues.Length);
                    cyclistTraining.SetModelData(initPriors);
                    zPosteriors = cyclistTraining.InferModelData();

                    for (int j = 0; j < n; j++)
                    {
                        zslowValues[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                    }
                }

                slowPrevNat = slowNoiseTuple.Item2;
                //------slow model end------

                packetCounter = N + 1;

                //------ideal model------
                double[] zValuesIdeal = new double[n];
                double[] idealGammas = new double[n];

                for (int i = 0; i < n; i++)
                {
                    zValuesIdeal[i] = 0.5;
                }
                for (int j = 0; j < n; j++)
                {
                    idealGammas[j] = precisions[bitCounter + j, 0];
                }

                zValuesIdeal = ZResponsibilities(idealGammas, shiftedNoisyBits);
                initPriors = new ModelData(zValuesIdeal, zValuesIdeal.Length);
                idealmodel.SetModelData(initPriors);
                zPosteriors = idealmodel.InferModelData();

                for (int j = 0; j < n; j++)
                {
                    zValuesIdeal[j] = zPosteriors.vBernoulliDistArray[j].GetProbTrue();
                }
                //------ideal model end------

                var storeEstPrec = new StringBuilder();

                for (int i = 0; i < n; i++)
                {
                    var newLine = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}",
                        data[N, i],
                        precisions[bitCounter + i, 0],
                        shiftedNoisyBits[i],
                        zfastValues[i],
                        fastAlpha,
                        fastBeta,
                        zslowValues[i],
                        slowAlpha,
                        slowBeta,
                        zValuesIdeal[i]);
                    storeEstPrec.AppendLine(newLine);
                }

                File.AppendAllText("simple-test.csv", storeEstPrec.ToString());
            }

            Console.WriteLine("Bitcounter: {0}", bitCounter);
            Console.WriteLine("--------------DONE---------------");

        }

    }

}