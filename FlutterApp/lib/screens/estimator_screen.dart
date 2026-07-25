import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../constants/app_constants.dart';
import '../models/car_input_model.dart';
import '../services/api_service.dart';
import '../widgets/custom_dropdown.dart';

class EstimatorScreen extends StatefulWidget {
  const EstimatorScreen({super.key});

  @override
  State<EstimatorScreen> createState() => _EstimatorScreenState();
}

class _EstimatorScreenState extends State<EstimatorScreen> {
  final _formKey = GlobalKey<FormState>();
  final ApiService _apiService = ApiService();

  // State parameter allocations
  final TextEditingController _yearController = TextEditingController();
  final TextEditingController _mileageController = TextEditingController();
  final TextEditingController _engineController = TextEditingController();
  final TextEditingController _hpController = TextEditingController();

  String _selectedBrand = AppConstants.brands.first;
  String _selectedTransmission = AppConstants.transmissions.first;
  String _selectedFuel = AppConstants.fuelTypes.first;

  bool _isLoading = false;
  double? _calculatedPrice;
  String? _errorMessage;

  final NumberFormat _currencyFormat = NumberFormat.currency(symbol: '\$', decimalDigits: 2);

  @override
  void dispose() {
    _yearController.dispose();
    _mileageController.dispose();
    _engineController.dispose();
    _hpController.dispose();
    super.dispose();
  }

  Future<void> _submitEvaluationRequest() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _calculatedPrice = null;
    });

    final inputData = CarInputModel(
      year: int.parse(_yearController.text),
      mileage: double.parse(_mileageController.text),
      engineSize: double.parse(_engineController.text),
      hp: double.parse(_hpController.text),
      brand: _selectedBrand,
      transmission: _selectedTransmission,
      fuelType: _selectedFuel,
    );

    try {
      final price = await _apiService.fetchCarValuation(inputData);
      setState(() {
        _calculatedPrice = price;
      });
    } catch (error) {
      setState(() {
        _errorMessage = error.toString().replaceAll('HttpException: ', '');
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Used Car Asset Valuation Engine', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        elevation: 2,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Card(
                  elevation: 0,
                  color: Colors.blue[50],
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      children: [
                        Icon(Icons.analytics, color: Colors.blue[800], size: 36),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Text(
                            'Provide vehicle structural metrics below to query remote production predictive inference frameworks.',
                            style: TextStyle(color: Colors.blue[900], fontSize: 13),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                CustomDropdown(
                  labelText: 'Manufacturer Brand',
                  value: _selectedBrand,
                  items: AppConstants.brands,
                  onChanged: (val) => setState(() => _selectedBrand = val!),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: TextFormField(
                        controller: _yearController,
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          labelText: 'Model Year',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) return 'Required';
                          final y = int.tryParse(v);
                          if (y == null || y < 1980 || y > 2027) return '1980 - 2027';
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TextFormField(
                        controller: _mileageController,
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          labelText: 'Odometer (Miles)',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) return 'Required';
                          final m = double.tryParse(v);
                          if (m == null || m < 0) return 'Invalid';
                          return null;
                        },
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: TextFormField(
                        controller: _engineController,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                        decoration: InputDecoration(
                          labelText: 'Engine Size (Liters)',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) return 'Required';
                          final e = double.tryParse(v);
                          if (e == null || e <= 0 || e > 10.0) return '0.1 - 10.0L';
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TextFormField(
                        controller: _hpController,
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          labelText: 'Horsepower',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) return 'Required';
                          final h = double.tryParse(v);
                          if (h == null || h <= 0) return 'Invalid';
                          return null;
                        },
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                CustomDropdown(
                  labelText: 'Gearbox Transmission',
                  value: _selectedTransmission,
                  items: AppConstants.transmissions,
                  onChanged: (val) => setState(() => _selectedTransmission = val!),
                ),
                const SizedBox(height: 16),
                CustomDropdown(
                  labelText: 'Combustible Fuel Type',
                  value: _selectedFuel,
                  items: AppConstants.fuelTypes,
                  onChanged: (val) => setState(() => _selectedFuel = val!),
                ),
                const SizedBox(height: 32),
                ElevatedButton(
                  onPressed: _isLoading ? null : _submitEvaluationRequest,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue[800],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Text('Compute Asset Inferences', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
                const SizedBox(height: 24),
                if (_errorMessage != null)
                  Card(
                    color: Colors.red[50],
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(_errorMessage!, style: TextStyle(color: Colors.red[900], fontWeight: FontWeight.w600)),
                    ),
                  ),
                if (_calculatedPrice != null)
                  Card(
                    elevation: 4,
                    color: Colors.green[800],
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Column(
                        children: [
                          const Text('ESTIMATED MARKET VALUATION', style: TextStyle(color: Colors.white70, fontSize: 12, letterSpacing: 1.2)),
                          const SizedBox(height: 8),
                          Text(
                            _currencyFormat.format(_calculatedPrice),
                            style: const TextStyle(color: Colors.white, fontSize: 32, fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}