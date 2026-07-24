class AppConstants {
  static const String apiBaseUrl = 'https://used-car-pricing-api.onrender.com';
  static const String predictEndpoint = '$apiBaseUrl/predict';
  
  static const List<String> brands = [
    'Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Chevrolet'
  ];
  
  static const List<String> transmissions = ['Automatic', 'Manual', 'CVT'];
  static const List<String> fuelTypes = ['Petrol', 'Diesel', 'Electric', 'Hybrid'];
}