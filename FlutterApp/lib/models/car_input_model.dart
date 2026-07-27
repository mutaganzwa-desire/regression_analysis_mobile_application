class CarInputModel {
  final int year;
  final double mileage;
  final double engineSize;
  final double hp;
  final String brand;
  final String transmission;
  final String fuelType;

  CarInputModel({
    required this.year,
    required this.mileage,
    required this.engineSize,
    required this.hp,
    required this.brand,
    required this.transmission,
    required this.fuelType,
  });

  Map<String, dynamic> toJson() {
    return {
      'brand': brand,
      'model_year': year,
      'milage': mileage,           // FastAPI schema expects 'milage'
      'engine_size': engineSize,
      'horsepower': hp.toInt(),    // Send as int to match schema
      'transmission': transmission,
      'fuel_type': fuelType,
      'accident': 'None reported', // Default value so FastAPI validation succeeds
    };
  }
}