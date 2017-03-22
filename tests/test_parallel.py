import numpy as np
import unittest
import scipy.interpolate
import os
import shutil


from uncertainpy import Parallel

from testing_classes import TestingFeatures
from testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from testing_classes import TestingModelNoTime, TestingModelNoTimeU
from testing_classes import TestingModelAdaptive, TestingModelConstant
from testing_classes import TestingModelNewProcess




class TestParallel(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.features = TestingFeatures(features_to_run=["feature0d",
                                                         "feature1d",
                                                         "feature2d",
                                                         "featureInvalid",
                                                         "feature_adaptive"])
        self.parallel = Parallel(model=TestingModel1d(),
                                 features=self.features)

        self.model_parameters = {"a": 0, "b": 1}

        self.t = np.arange(0, 10)
        self.U = np.arange(0, 10) + 1


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)




    def test_init(self):
        Parallel(TestingModel1d())


    def test_feature(self):
        self.parallel.features = 1
        self.assertEqual(self.parallel.features, 1)


    def test_model(self):
        self.parallel.model = 1
        self.assertEqual(self.parallel.model, 1)


    def test_run_subprocess(self):
        t, U = self.parallel.run_subprocess(self.model_parameters)

        self.assertTrue(np.array_equal(self.t, t))
        self.assertTrue(np.array_equal(self.U, U))






    def test_sort_features(self):
        results = {"directComparison": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10)},
                   "feature1d": {"U": np.arange(0, 10),
                                 "t": np.arange(0, 10)},
                   "feature0d": {"U": 1,
                                 "t": None},
                   "feature2d": {"U": np.array([np.arange(0, 10),
                                                np.arange(0, 10)]),
                                 "t": np.arange(0, 10)},
                   "feature_adaptive": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10),
                                        "interpolation": "interpolation object"},
                   "featureInvalid": {"U": None,
                                      "t": None}}

        features_0d, features_1d, features_2d = self.parallel.sortFeatures(results)

        self.assertEqual(features_0d, ["feature0d", "featureInvalid"])
        self.assertEqual(set(features_1d),
                         set(["feature1d", "directComparison", "feature_adaptive"]))
        self.assertEqual(features_2d, ["feature2d"])




    def test_createInterpolations(self):
        results = {"directComparison": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10)},
                   "feature1d": {"U": np.arange(0, 10),
                                 "t": np.arange(0, 10)},
                   "feature0d": {"U": 1,
                                 "t": None},
                   "feature2d": {"U": np.array([np.arange(0, 10),
                                                np.arange(0, 10)]),
                                 "t": np.arange(0, 10)},
                   "feature_adaptive": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10)},
                   "featureInvalid": {"U": None,
                                      "t": None}}

        results = self.parallel.createInterpolations(results)

        self.assertTrue(np.array_equal(results["directComparison"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["directComparison"]["U"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["U"], np.arange(0, 10)))
        self.assertIsNone(results["feature0d"]["t"])
        self.assertEqual(results["feature0d"]["U"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["U"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertIsNone(results["featureInvalid"]["t"])
        self.assertIsNone(results["featureInvalid"]["U"])
        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)



    def test_createInterpolations_1d_no_t(self):
        results = {"feature_adaptive": {"U": np.arange(0, 10),
                                        "t": None}}

        with self.assertRaises(AttributeError):
            self.parallel.createInterpolations(results)


    def test_createInterpolations_0d(self):
        results = {"feature_adaptive": {"U": 1,
                                        "t": np.arange(0, 10)}}

        with self.assertRaises(AttributeError):
            self.parallel.createInterpolations(results)


    def test_createInterpolations_2d(self):
        results = {"feature_adaptive": {"U": np.array([np.arange(0, 10),
                                                       np.arange(0, 10)]),
                                        "t": np.arange(0, 10)}}

        with self.assertRaises(NotImplementedError):
            self.parallel.createInterpolations(results)


    def test_run(self):
        results = self.parallel.run(self.model_parameters)


        self.assertTrue(np.array_equal(results["directComparison"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["directComparison"]["U"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["U"], np.arange(0, 10)))
        self.assertIsNone(results["feature0d"]["t"])
        self.assertEqual(results["feature0d"]["U"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["U"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertIsNone(results["featureInvalid"]["t"])
        self.assertIsNone(results["featureInvalid"]["U"])
        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_run_adaptive_model(self):
        parallel = Parallel(model=TestingModelAdaptive(),
                            features=TestingFeatures(features_to_run="feature_adaptive"))
        results = parallel.run(self.model_parameters)

        self.assertTrue(np.array_equal(results["directComparison"]["t"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["directComparison"]["U"], np.arange(0, 11) + 1))
        self.assertIsInstance(results["directComparison"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 11) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_run_model_no_time(self):
        parallel = Parallel(model=TestingModelNoTime())
        with self.assertRaises(RuntimeError):
            parallel.run(self.model_parameters)


    def test_run_feature_no_time(self):
        parallel = Parallel(model=TestingModel1d(),
                            features=TestingFeatures(features_to_run="feature_no_time"))

        with self.assertRaises(RuntimeError):
            parallel.run(self.model_parameters)