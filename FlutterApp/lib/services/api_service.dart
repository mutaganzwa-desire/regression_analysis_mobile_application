import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../constants/app_constants.dart';
import '../models/car_input_model.dart';

class ApiService {
  final http.Client client;

  ApiService({http.Client? client}) : this.client = client ?? http.Client();

  Future<double> fetchCarValuation(CarInputModel data) async {
    try {
      final response = await client.post(
        Uri.parse(AppConstants.predictEndpoint),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(data.toJson()),
      ).timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseBody = jsonDecode(response.body);
        return (responseBody['predicted_price'] as num).toDouble();
      } else {
        final Map<String, dynamic> errorData = jsonDecode(response.body);
        
        // Safely parse detail if it's a String or List
        String detailMessage = 'Remote platform verification error.';
        if (errorData['detail'] != null) {
          if (errorData['detail'] is String) {
            detailMessage = errorData['detail'];
          } else if (errorData['detail'] is List) {
            final List errors = errorData['detail'];
            detailMessage = errors.map((e) => "${e['loc']?.last}: ${e['msg']}").join(', ');
          }
        }
        
        throw HttpException(detailMessage);
      }
    } on SocketException {
      throw const HttpException('Network connection timeout. Check connectivity links.');
    } catch (e) {
      throw HttpException('Processing Exception: ${e.toString()}');
    }
  }
}