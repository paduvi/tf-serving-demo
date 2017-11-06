package com.paduvi;

import java.util.Arrays;

import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import org.tensorflow.framework.TensorShapeProto.Dim;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import tensorflow.serving.PredictionServiceGrpc;
import tensorflow.serving.PredictionServiceGrpc.PredictionServiceBlockingStub;
import tensorflow.serving.Model.ModelSpec;
import tensorflow.serving.Predict.PredictRequest;

/**
 * Hello world!
 *
 */
public class App {

	static final String HOST = "103.56.158.245";

	public static PredictRequest createPredictRequest(double x[], double y[]) {
		ModelSpec modelSpec = ModelSpec.newBuilder().setName("default").setSignatureName("magic_model").build();
		PredictRequest.Builder builder = PredictRequest.newBuilder().setModelSpec(modelSpec)
				.putInputs("egg", createTensor(x)).putInputs("bacon", createTensor(y));
		return builder.build();
	}

	public static TensorProto createTensor(double values[]) {
		Dim dim = Dim.newBuilder().setSize(values.length).build();
		TensorShapeProto shape = TensorShapeProto.newBuilder().addDim(dim).build();
		TensorProto.Builder builder = TensorProto.newBuilder().setDtype(DataType.DT_FLOAT).setTensorShape(shape);
		for (double value : values) {
			builder.addFloatVal((float) value);
		}
		return builder.build();
	}

	public static void main(String[] args) {
		ManagedChannel channel = ManagedChannelBuilder.forAddress(HOST, 8500).usePlaintext(true).build();

		PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
		double[] x = new double[]{1.0, 6.0};
		double[] y = new double[]{18.0, 107.0};
		Float[] z = stub.predict(createPredictRequest(x, y)).getOutputsOrThrow("spam").getFloatValList().toArray(new Float[0]);
		System.out.println(Arrays.toString(z));
	}
}
