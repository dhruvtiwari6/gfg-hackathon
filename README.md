# IRCTC Ticket Booker

An automated script to help book train tickets on IRCTC from Kota to Ludhiana.

## Features

- 🚀 Automated browser setup with Playwright
- 🔐 Secure login with environment variables
- 🚄 Train search and selection
- 👤 Passenger details management
- 💳 Payment process initiation
- ⚙️ Configurable booking parameters

## Prerequisites

- Python 3.12+
- Playwright browser (will be installed automatically)
- IRCTC account credentials

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers:**
   ```bash
   playwright install
   ```

3. **Set up environment variables:**
   - Copy the `.env.example` file to `.env`
   - Fill in your IRCTC credentials and booking preferences

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# IRCTC Credentials
IRCTC_USERNAME=your_irctc_username
IRCTC_PASSWORD=your_irctc_password

# Booking Configuration
FROM_STATION=Kota
TO_STATION=Ludhiana
TRAVEL_DATE=2024-02-06  # Format: YYYY-MM-DD
PASSENGER_NAME=Your Name
PASSENGER_AGE=25
PASSENGER_GENDER=M
PASSENGER_BERTH=UB  # UB=Upper Berth, LB=Lower Berth, etc.
```

### Booking Parameters

- **FROM_STATION**: Departure station (e.g., "Kota")
- **TO_STATION**: Destination station (e.g., "Ludhiana")
- **TRAVEL_DATE**: Travel date in YYYY-MM-DD format
- **PASSENGER_NAME**: Full name of the passenger
- **PASSENGER_AGE**: Age of the passenger
- **PASSENGER_GENDER**: Gender (M/F/T)
- **PASSENGER_BERTH**: Berth preference (UB, LB, MB, etc.)

## Usage

### Basic Usage

```bash
python irctc_booker.py
```

### Advanced Configuration

You can modify the script to customize:

- Multiple passengers
- Specific train selection
- Different berth preferences
- Headless mode execution

### Example with Custom Configuration

```python
# In the main() function, modify the config:
config = {
    'username': os.getenv('IRCTC_USERNAME'),
    'password': os.getenv('IRCTC_PASSWORD'),
    'from_station': 'Kota Junction',
    'to_station': 'Ludhiana Junction',
    'date': '06/02/2024',  # DD/MM/YYYY format
    'train_number': '12345',  # Optional: specific train
    'passengers': [
        {
            'name': 'John Doe',
            'age': 30,
            'gender': 'M',
            'berth_preference': 'LB'  # Lower Berth
        },
        {
            'name': 'Jane Doe',
            'age': 28,
            'gender': 'F',
            'berth_preference': 'UB'  # Upper Berth
        }
    ]
}
```

## How It Works

1. **Browser Setup**: Launches a Playwright-controlled browser
2. **Login**: Navigates to IRCTC and logs in with your credentials
3. **Train Search**: Searches for trains between Kota and Ludhiana
4. **Train Selection**: Selects the first available train or a specific one
5. **Passenger Details**: Fills in passenger information
6. **Payment**: Initiates the payment process (manual completion required)

## Important Notes

⚠️ **Manual Intervention Required**:
- CAPTCHA solving during login
- Payment completion
- Final booking confirmation

🔒 **Security**:
- Never commit your `.env` file to version control
- Use strong, unique passwords for IRCTC
- Consider using a dedicated IRCTC account for automation

🤖 **Anti-Bot Measures**:
- The script includes measures to avoid detection
- Use responsibly and follow IRCTC terms of service
- Consider adding delays between actions if needed

## Troubleshooting

### Common Issues

1. **Login Failures**:
   - Check your credentials in `.env`
   - Ensure CAPTCHA is solved manually
   - Verify IRCTC website is accessible

2. **Train Search Failures**:
   - Check station names are correct
   - Verify travel date format
   - Ensure trains are available for the date

3. **Playwright Issues**:
   - Run `playwright install` to ensure browsers are installed
   - Check internet connection
   - Try running in headless mode

### Debug Mode

For debugging, you can:
- Set `headless=False` in `setup_browser()` to see browser actions
- Add `await self.page.wait_for_timeout(5000)` for manual inspection
- Check browser console for errors

## Legal Disclaimer

This script is for educational purposes only. Use it responsibly and in accordance with IRCTC's terms of service. The author is not responsible for any misuse or violations of terms of service.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all dependencies are installed
4. Verify your configuration

For Kota to Ludhiana routes, popular trains include:
- Shatabdi Express
- Rajdhani Express
- Various Mail/Express trains

Always check train availability and timings before booking.