#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>


// helper func to read big-endian(?) ints from file
int read_int(std::ifstream& file) {
	unsigned char bytes[4];
	file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
	return (int)((bytes[0] << 24 ) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

// func to load mnist images
std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("could not open file :( -- " + path);
	}

	// read headers
	int magic_number = read_int(file);
	int num_images = read_int(file);
	int rows = read_int(file);
	int cols = read_int(file);

	std::vector<std::vector<float>> images(num_images, std::vector<float>(rows * cols));

	for (int i = 0; i < num_images; ++i) {
		for (int j = 0; j < rows * cols; ++j) {
			unsigned char pixel = 0;
			file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			images[i][j] = pixel / 255.0f; // normalize pixel values to [0, 1]
		}
	}

	return images;
}

// func to load mnist labels
std::vector<int> load_mnist_labels(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("could not open file :( -- " + path);
	}

	//read headers
	int magic_number = read_int(file);
	int num_labels = read_int(file);

	std::vector<int> labels(num_labels);

	for (int i = 0; i < num_labels; ++i) {
		unsigned char label = 0;
		file.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels[i] = label;
	}

	return labels;
}

std::vector<float> one_hot_encode(int label, int num_classes) {
	std::vector<float> encoded(num_classes, 0.0f);
	if (label < num_classes) {
		encoded[label] = 1.0f;
	}
	return encoded;
}


class MLP {
public:
	MLP(int input_size, int hidden_size, int output_size)
		: input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
		initialize_weights_and_biases();
	}

	void initialize_weights_and_biases() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> dist(0.0, 1.0);

		W1.resize(input_size * hidden_size);
		W2.resize(hidden_size * output_size);
		b1.resize(hidden_size, 1.0);
		b2.resize(output_size, 1.0);

		// init weights w/ random vals
		for (auto& weight : W1) weight = dist(gen) * sqrt(2. / input_size);
		for (auto& weight : W2) weight = dist(gen) * sqrt(2. / hidden_size);
	}

	std::vector<float> forward(const std::vector<float>& inputs) {
		hidden_layer = matrix_multiply(inputs, W1, input_size, hidden_size);
		add_bias(hidden_layer, b1);
		apply_sigmoid(hidden_layer);

		output_layer = matrix_multiply(hidden_layer, W2, hidden_size, output_size);
		add_bias(output_layer, b2);
		apply_sigmoid(output_layer);

		return output_layer;
	}

	void backward(const std::vector<float>& inputs, const std::vector<float>& expected_outputs, float learning_rate) {
		// calc output layer error
		std::vector<float> output_errors = output_layer;
		for (size_t i = 0; i < output_layer.size(); ++i) {
			output_errors[i] -= expected_outputs[i];
			output_errors[i] *= output_layer[i] * (1 - output_layer[i]);
		}

		// calc hidden layer error
		std::vector<float> hidden_errors(hidden_size, 0.0);
		for (size_t j = 0; j < hidden_size; ++j) {
			for (size_t k = 0; k < output_size; ++k) {
				hidden_errors[j] += output_errors[k] * W2[j * output_size + k];
			}
			hidden_errors[j] *= hidden_layer[j] * (1 - hidden_layer[j]);
		}

		// update weights and biases for output layer
		for (size_t j = 0; j < hidden_size; ++j) {
			for (size_t k = 0; k < output_size; ++k) {
				W2[j * output_size + k] -= learning_rate * output_errors[k] * hidden_layer[j];
			}
		}
		for (size_t k = 0; k < output_size; ++k) {
			b2[k] -= learning_rate * output_errors[k];
		}

		// update weights and biases for hidden layer
		for (size_t i = 0; i < input_size; ++i) {
			for (size_t j = 0; j < hidden_size; ++j) {
				W1[i * hidden_size + j] -= learning_rate * hidden_errors[j] * inputs[i];
			}
		}
		for (size_t j = 0; j < hidden_size; ++j) {
			b1[j] -= learning_rate * hidden_errors[j];
		}
	}

	int predict(const std::vector<float>& inputs) {
		auto outputs = forward(inputs);
		return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
	}

private:
	int input_size, hidden_size, output_size;
	std::vector<float> W1, W2, b1, b2;
	std::vector<float> hidden_layer, output_layer;

	std::vector<float> matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, int a_cols, int b_cols) {
		std::vector<float> result(a.size() / a_cols * b_cols, 0.0);
		for (int i = 0; i < static_cast<int>(a.size() / a_cols); ++i) {
			for (int j = 0; j < b_cols; ++j) {
				for (int k = 0; k < a_cols; ++k) {
					result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
				}
			}
		}
		return result;
	}

	void add_bias(std::vector<float>& a, const std::vector<float>& b) {
		for (size_t i = 0; i < a.size(); i++) {
			a[i] += b[i % b.size()];
		}
	}

	void apply_sigmoid(std::vector<float>& a) {
		for (auto& val : a) val = 1.0 / (1.0 + std::exp(-val));
	}
};


int main() {
	try {
		std::string base_path = "./MNIST_data/";
		std::string train_images_path = base_path + "train-images.idx3-ubyte";
		std::string train_labels_path = base_path + "train-labels.idx1-ubyte";
		std::string test_images_path = base_path + "t10k-images.idx3-ubyte";
		std::string test_labels_path = base_path + "t10k-labels.idx1-ubyte";

		std::cout << "loading training data...\n";
		auto train_images = load_mnist_images(train_images_path);
		auto train_labels = load_mnist_labels(train_labels_path);
		std::cout << "loading testing data...\n";
		auto test_images = load_mnist_images(test_images_path);
		auto test_labels = load_mnist_labels(test_labels_path);

		// params
		int input_size = 784;
		int hidden_size = 128;
		int output_size = 10;
		float learning_rate = 0.1;
		int epochs = 10;

		MLP mlp(input_size, hidden_size, output_size);

		// train loop
		std::cout << "starting training...\n";
		for (int epoch = 0; epoch < epochs; ++epoch) {
			std::cout << "epoch " << (epoch + 1) << "/" << epochs << "\n";
			for (size_t i = 0; i < train_images.size(); ++i) {
				std::vector<float> expected_output = one_hot_encode(train_labels[i], output_size);
				mlp.forward(train_images[i]);
				mlp.backward(train_images[i], expected_output, learning_rate);
			}
		}

		// test loop
		std::cout << "testing model..\n";
		int correct = 0;
		for (size_t i = 0; i < test_images.size(); ++i) {
			int prediction = mlp.predict(test_images[i]);
			if (prediction == test_labels[i]) {
				++correct;
			}
		}

		double accuracy = 100.0 * correct / test_images.size();
		std::cout << "Accuracy: " << accuracy << "%\n";
	
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

