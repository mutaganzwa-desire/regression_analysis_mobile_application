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
      'year': year,
      'mileage': mileage,
      'engine_size': engineSize,
      'hp': hp,
      'brand': brand,
      'transmission': transmission,
      'fuel_type': fuelType,
    };
  }
}