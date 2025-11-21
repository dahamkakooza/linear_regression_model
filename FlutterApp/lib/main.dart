import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const CropYieldPredictorApp());
}

class CropYieldPredictorApp extends StatelessWidget {
  const CropYieldPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Yield Predictor',
      theme: ThemeData(
        primarySwatch: Colors.green,
        useMaterial3: true,
      ),
      home: const PredictionPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  // Your deployed API endpoint
  final String apiUrl = "https://crop-yield-api-pfsb.onrender.com/predict";

  // Text editing controllers for input fields
  final TextEditingController nController = TextEditingController(text: "90");
  final TextEditingController pController = TextEditingController(text: "42");
  final TextEditingController kController = TextEditingController(text: "43");
  final TextEditingController tempController = TextEditingController(text: "25");
  final TextEditingController humidityController = TextEditingController(text: "82");
  final TextEditingController phController = TextEditingController(text: "6.5");
  final TextEditingController rainfallController = TextEditingController(text: "203");

  // Crop selection
  String selectedCrop = "rice";

  // Available crops from your API
  final List<String> availableCrops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
  ];

  // State variables
  String predictionResult = '';
  bool isLoading = false;
  String errorMessage = '';

  Future<void> predictYield() async {
    // Validate inputs first
    if (!_validateInputs()) {
      return;
    }

    setState(() {
      isLoading = true;
      errorMessage = '';
      predictionResult = '';
    });

    try {
      // Prepare the request body
      final Map<String, dynamic> requestBody = {
        "N": double.parse(nController.text),
        "P": double.parse(pController.text),
        "K": double.parse(kController.text),
        "temperature": double.parse(tempController.text),
        "humidity": double.parse(humidityController.text),
        "ph": double.parse(phController.text),
        "rainfall": double.parse(rainfallController.text),
        "crop": selectedCrop,
      };

      print("🌱 Making API call to: $apiUrl");
      print("📦 Request data: $requestBody");

      // Make API call with timeout
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: json.encode(requestBody),
      ).timeout(const Duration(seconds: 30));

      print("📡 Response status: ${response.statusCode}");
      print("📡 Response body: ${response.body}");

      if (response.statusCode == 200) {
        // Success - parse the response
        final data = json.decode(response.body);
        setState(() {
          predictionResult = '${data['predicted_yield_kg_ha']} kg/ha';
        });
        print("✅ Prediction successful: $predictionResult");
      } else {
        // API returned error
        final errorData = json.decode(response.body);
        setState(() {
          errorMessage = 'API Error: ${errorData['detail'] ?? 'Unknown error (Status: ${response.statusCode})'}';
        });
        print("❌ API error: $errorMessage");
      }
    } catch (e) {
      // Network or other errors
      setState(() {
        errorMessage = 'Connection error: $e\n\nPlease check your internet connection and ensure the API is running.';
      });
      print("❌ Connection error: $e");
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  bool _validateInputs() {
    // Check if all fields are filled with valid numbers
    final controllers = [
      nController, pController, kController, tempController,
      humidityController, phController, rainfallController
    ];

    for (var controller in controllers) {
      if (controller.text.isEmpty) {
        setState(() {
          errorMessage = 'Please fill in all fields';
        });
        return false;
      }

      final value = double.tryParse(controller.text);
      if (value == null) {
        setState(() {
          errorMessage = 'Please enter valid numbers in all fields';
        });
        return false;
      }
    }

    // Validate ranges
    final n = double.parse(nController.text);
    final p = double.parse(pController.text);
    final k = double.parse(kController.text);
    final temp = double.parse(tempController.text);
    final humidity = double.parse(humidityController.text);
    final ph = double.parse(phController.text);
    final rainfall = double.parse(rainfallController.text);

    if (n < 0 || n > 140) {
      setState(() { errorMessage = 'Nitrogen (N) must be between 0-140'; });
      return false;
    }
    if (p < 5 || p > 145) {
      setState(() { errorMessage = 'Phosphorus (P) must be between 5-145'; });
      return false;
    }
    if (k < 5 || k > 205) {
      setState(() { errorMessage = 'Potassium (K) must be between 5-205'; });
      return false;
    }
    if (temp < 8 || temp > 43) {
      setState(() { errorMessage = 'Temperature must be between 8-43°C'; });
      return false;
    }
    if (humidity < 14 || humidity > 100) {
      setState(() { errorMessage = 'Humidity must be between 14-100%'; });
      return false;
    }
    if (ph < 3.5 || ph > 9.5) {
      setState(() { errorMessage = 'pH must be between 3.5-9.5'; });
      return false;
    }
    if (rainfall < 20 || rainfall > 300) {
      setState(() { errorMessage = 'Rainfall must be between 20-300mm'; });
      return false;
    }

    return true;
  }

  Widget _buildInputField(String label, TextEditingController controller, String range) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.green,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              decoration: InputDecoration(
                hintText: "Range: $range",
                border: const OutlineInputBorder(),
                contentPadding: const EdgeInsets.symmetric(horizontal: 12),
              ),
              keyboardType: TextInputType.number,
              style: const TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Crop Yield Predictor',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header Card
            Card(
              elevation: 4,
              color: Colors.green[50],
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    const Icon(
                      Icons.agriculture,
                      size: 50,
                      color: Colors.green,
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Crop Yield Prediction',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.green,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Connected to: ${apiUrl.split('/predict')[0]}',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.grey,
                        fontStyle: FontStyle.italic,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Input Section
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Soil & Environmental Parameters',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.green,
                      ),
                    ),
                    const SizedBox(height: 20),

                    // Soil Nutrients
                    const Text(
                      'Soil Nutrients',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    _buildInputField('Nitrogen (N)', nController, '0-140'),
                    _buildInputField('Phosphorus (P)', pController, '5-145'),
                    _buildInputField('Potassium (K)', kController, '5-205'),

                    const SizedBox(height: 20),

                    // Environmental Factors
                    const Text(
                      'Environmental Factors',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    _buildInputField('Temperature (°C)', tempController, '8-43'),
                    _buildInputField('Humidity (%)', humidityController, '14-100'),
                    _buildInputField('pH Level', phController, '3.5-9.5'),
                    _buildInputField('Rainfall (mm)', rainfallController, '20-300'),

                    const SizedBox(height: 20),

                    // Crop Selection
                    const Text(
                      'Crop Selection',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Card(
                      elevation: 2,
                      child: Padding(
                        padding: const EdgeInsets.all(12.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Select Crop Type',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            const SizedBox(height: 8),
                            DropdownButtonFormField<String>(
                              value: selectedCrop,
                              decoration: const InputDecoration(
                                border: OutlineInputBorder(),
                                contentPadding: EdgeInsets.symmetric(horizontal: 12),
                              ),
                              items: availableCrops.map((String crop) {
                                return DropdownMenuItem<String>(
                                  value: crop,
                                  child: Text(
                                    crop[0].toUpperCase() + crop.substring(1),
                                    style: const TextStyle(fontSize: 16),
                                  ),
                                );
                              }).toList(),
                              onChanged: (String? newValue) {
                                setState(() {
                                  selectedCrop = newValue!;
                                });
                              },
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Predict Button
            SizedBox(
              width: double.infinity,
              height: 60,
              child: ElevatedButton(
                onPressed: isLoading ? null : predictYield,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: isLoading
                    ? const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    ),
                    SizedBox(width: 12),
                    Text(
                      'Predicting...',
                      style: TextStyle(fontSize: 18),
                    ),
                  ],
                )
                    : const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.analytics, size: 24),
                    SizedBox(width: 8),
                    Text(
                      'PREDICT YIELD',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Results Section
            if (predictionResult.isNotEmpty)
              Card(
                elevation: 4,
                color: Colors.green[50],
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.emoji_events,
                        size: 48,
                        color: Colors.green,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Predicted Yield',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        predictionResult,
                        style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'Based on Random Forest Model',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.green,
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // Error Message
            if (errorMessage.isNotEmpty)
              Card(
                elevation: 4,
                color: Colors.red[50],
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      const Icon(Icons.error, color: Colors.red),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          errorMessage,
                          style: const TextStyle(color: Colors.red),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    // Clean up all controllers
    nController.dispose();
    pController.dispose();
    kController.dispose();
    tempController.dispose();
    humidityController.dispose();
    phController.dispose();
    rainfallController.dispose();
    super.dispose();
  }
}