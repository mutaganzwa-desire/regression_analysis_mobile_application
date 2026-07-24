import 'package:flutter/material.dart';
import 'screens/estimator_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CarEstimatorApp());
}

class CarEstimatorApp extends StatelessWidget {
  const CarEstimatorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Car Estimator System',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.light,
        ),
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Color(0xFFFAFAFA),
        ),
      ),
      home: const EstimatorScreen(),
    );
  }
}