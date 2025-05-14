import os
import matplotlib

# Set the backend to Agg before importing pyplot
matplotlib.use("Agg")

from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from sklearn.cluster import KMeans  # Import KMeans
from collections import defaultdict  # Import defaultdict


app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a random secret key

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1624  # Limit max upload size to 16 MB

# Configure static folder for plots
PLOT_FOLDER = "static/plots/"
app.config["PLOT_FOLDER"] = PLOT_FOLDER


# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Your MySQL username
        password="Tharun@2000",  # Your MySQL password
        database="flask_app_db",
    )


# Function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_coordinates(address):
    """
    Geocodes an address string to latitude and longitude.
    Uses the geopy library and the Nominatim geocoder.
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut

    geolocator = Nominatim(user_agent="issue_reporting_app")  # Specify a user agent
    try:
        location = geolocator.geocode(address, timeout=10)  # Increase timeout
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        print("Geocoding timed out for address:", address)
        return None, None
    except Exception as e:
        print("Geocoding error:", e)
        return None, None


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone_number = request.form["phone_number"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE phone_number = %s", (phone_number,))
        user = cursor.fetchone()

        if user and check_password_hash(user[4], password):
            session["user_id"] = user[0]
            session["name"] = user[1]
            flash("Login successful!", "success")
            cursor.close()
            conn.close()
            return redirect(url_for("user_form"))

        cursor.execute("SELECT * FROM admins WHERE phone_number = %s", (phone_number,))
        admin = cursor.fetchone()

        if admin and admin[4] == password:
            session["admin_id"] = admin[0]
            flash("Admin login successful!", "success")
            cursor.close()
            conn.close()
            return redirect(url_for("admin_page"))

        flash("Invalid phone number or password.", "danger")
        cursor.close()
        conn.close()

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        phone_number = request.form["phone_number"]
        aadhaar_number = request.form["aadhaar_number"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (name, phone_number, aadhaar_number, password) VALUES (%s, %s, %s, %s)",
                (name, phone_number, aadhaar_number, hashed_password),
            )
            conn.commit()

            # Fetch the user details after successful signup
            cursor.execute(
                "SELECT * FROM users WHERE phone_number = %s", (phone_number,)
            )
            new_user = cursor.fetchone()

            session["user_id"] = new_user[0]
            session["name"] = new_user[1]

            flash("Registration successful! You are now logged in.", "success")

            cursor.close()
            conn.close()

            # Redirect to user_form with user details
            return redirect(
                url_for(
                    "user_form",
                    name=name,
                    phone_number=phone_number,
                    aadhaar_number=aadhaar_number,
                )
            )

        except mysql.connector.Error as err:
            flash(f"Error: {err}", "danger")
            return render_template(
                "signup.html"
            )  # Render the signup form again in case of error

        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    return render_template("signup.html")


@app.route("/user-form", methods=["GET", "POST"])
def user_form():
    if "user_id" not in session:
        return redirect(url_for("login"))

    name = request.args.get("name", session.get("name", ""))
    phone_number = request.args.get("phone_number", "")
    aadhaar_number = request.args.get("aadhaar_number", "")

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == "POST":
        option = request.form["option"]
        description = request.form["desc"]
        area = request.form["area"]
        latitude = request.form.get("latitude")  # Get Latitude from form
        longitude = request.form.get("longitude")  # Get Longitude from form

        file = request.files.get("photo")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            cursor.execute(
                "INSERT INTO userforms (user_id, name, phone_number, aadhaar_number, issue_option, `desc`, area, latitude, longitude) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session["user_id"],
                    name,
                    phone_number,
                    aadhaar_number,
                    option,
                    description,
                    area,
                    latitude,
                    longitude,
                ),
            )
            conn.commit()

            flash(
                "Your submission has been received successfully! Thank you for reporting.",
                "success",
            )

    cursor.close()
    conn.close()

    return render_template(
        "user_form.html",
        name=name,
        phone_number=phone_number,
        aadhaar_number=aadhaar_number,
    )


# Delete Userform
@app.route(
    "/delete_userform/<int:form_id>", methods=["POST"]
)  # Need to use POST since it's a delete operation
def delete_userform(form_id):
    if "admin_id" not in session:
        flash("Unauthorized access.", "danger")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM userforms WHERE id = %s", (form_id,))
        conn.commit()
        flash("Userform entry deleted successfully.", "success")
    except mysql.connector.Error as err:
        flash(f"Error deleting userform entry: {err}", "danger")
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for("admin_page"))  # Return to the same page


def generate_and_save_plots():
    """Generates and saves the plots to static files."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT issue_option, area FROM userforms")
    data = cursor.fetchall()

    if not data:
        return None, None, None  #  Kmeans Filename

    issues_count_dict = {}

    # Count issues reported by type.
    for issue_option, _ in data:
        if issue_option not in issues_count_dict:
            issues_count_dict[issue_option] = 0
        issues_count_dict[issue_option] += 1

    options = list(issues_count_dict.keys())

    # Create directory if it doesn't exist
    os.makedirs(app.config["PLOT_FOLDER"], exist_ok=True)

    # -----------------------------------------------------
    # Histogram
    plt.figure(figsize=(12, 6))  # Adjust figure size

    plt.bar(options, issues_count_dict.values())
    plt.title("Histogram of Issues Reported", fontsize=14)  # Adjust title size
    plt.xlabel("Issues", fontsize=12)  # Adjust label size
    plt.ylabel("Frequency", fontsize=12)  # Adjust y-tick font size
    plt.xticks(
        rotation=45, ha="right", fontsize=10
    )  # Adjust x-tick font size and rotation
    plt.yticks(fontsize=10)  # Adjust y-tick font size

    histogram_filename = "histogram.png"
    histogram_filepath = os.path.join(app.config["PLOT_FOLDER"], histogram_filename)
    plt.savefig(histogram_filepath)  # Save to file
    plt.close()  # Close the figure

    # -----------------------------------------------------
    # Heatmap (Revised for Better Visuals)

    # Prepare data for the heatmap
    issue_area_counts = {}
    for issue, area in data:
        if (issue, area) not in issue_area_counts:
            issue_area_counts[(issue, area)] = 0
        issue_area_counts[(issue, area)] += 1

    unique_issues = sorted(list(set([issue for issue, _ in data])))
    unique_areas = sorted(list(set([area for _, area in data])))

    heatmap_data = np.zeros((len(unique_areas), len(unique_issues)))
    for i, area in enumerate(unique_areas):
        for j, issue in enumerate(unique_issues):
            heatmap_data[i, j] = issue_area_counts.get((issue, area), 0)

    plt.figure(
        figsize=(12, 8)
    )  # Increased figure size for better readability, can also make it Dynamic for more data
    ax = sns.heatmap(
        heatmap_data,
        annot=True,  # Show count values
        fmt="g",  # Integer format
        cmap="YlGnBu",  # Use a sequential color palette
        xticklabels=unique_issues,
        yticklabels=unique_areas,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"shrink": 0.8},
    )
    plt.xticks(
        rotation=45, ha="right"
    )  # Rotate x-axis labels to prevent overlap, can also shorten the X and Y values
    plt.yticks(rotation=0)
    plt.title("Severity Prediction by Issue and Place", fontsize=16)
    plt.xlabel("Issue", fontsize=12)
    plt.ylabel("Area", fontsize=12)

    heatmap_filename = "heatmap.png"
    heatmap_filepath = os.path.join(app.config["PLOT_FOLDER"], heatmap_filename)
    plt.savefig(heatmap_filepath)
    plt.close()

    # -----------------------------------------------------
    # K-means plot
    cursor.execute(
        "SELECT latitude, longitude, area FROM userforms WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    )
    location_data = cursor.fetchall()

    if location_data:
        # Count occurrences of each location
        location_counts = defaultdict(int)
        location_area = {}  # Store the area associated with each location
        for lat, lon, area in location_data:
            location = (lat, lon)
            location_counts[location] += 1
            location_area[location] = area  # Store the area for this location

        # Prepare data for plotting
        unique_locations = list(location_counts.keys())
        locations = np.array(unique_locations)

        # Perform K-means clustering with a fixed number of clusters (3)
        n_clusters = min(
            3, len(locations)
        )  # Ensure we don't ask for more clusters than data points
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(locations)

        # Define colors for the clusters
        colors = ["red", "blue", "green"]  # Use only 3 colors
        cluster_colors = [
            colors[i % len(colors)] for i in clusters
        ]  # Cycle through colors

        # Plotting the clusters
        plt.figure(figsize=(12, 10))  # **Increased Figure Size for K-means**

        # Change the Subplot
        plt.subplot(4, 1, (4))  # Changed from plt.subplot(4, 1, 4)

        # Plot the scatter plot
        plt.scatter(
            locations[:, 1], locations[:, 0], c=cluster_colors, marker="o"
        )  # Use circle marker

        # Add count annotations
        for i, location in enumerate(unique_locations):
            count = location_counts[location]
            area = location_area[location]  # Get area for this location
            plt.annotate(
                f"{area} ({count})",
                (locations[i, 1], locations[i, 0]),
                fontsize=10,  # **Increased Annotation Font Size**
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )  # Reduced font size, added offset, and ha/va

        # Get the min and max latitude and longitude from the data
        min_lat = min(float(lat) for lat, lon in unique_locations)
        max_lat = max(float(lat) for lat, lon in unique_locations)
        min_lon = min(float(lon) for lat, lon in unique_locations)
        max_lon = max(float(lon) for lat, lon in unique_locations)

        # **Modified the Axis Limits to show original values**
        plt.xlim(min_lon, max_lon)
        plt.ylim(min_lat, max_lat)

        plt.xlabel("Longitude", fontsize=12)  # **Increased Axis Label Font Size**
        plt.ylabel("Latitude", fontsize=12)  # **Increased Axis Label Font Size**
        plt.title(
            "K-means Clustering of Issue Locations with Report Counts", fontsize=14
        )  # **Increased Title Font Size**

        # **Explicitly set the tick formatting to decimal notation and Round Ticks**
        plt.ticklabel_format(useOffset=False, style="plain", axis="both")

        # Determine appropriate tick intervals
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        lat_interval = 0.02  # Default interval for Latitude
        lon_interval = 0.05  # Default interval for Longitude

        if lat_range < 0.1:
            lat_interval = 0.01
        elif lat_range < 0.2:
            lat_interval = 0.02
        elif lat_range < 0.5:
            lat_interval = 0.05
        else:
            lat_interval = 0.1

        if lon_range < 0.1:
            lon_interval = 0.01
        elif lon_range < 0.2:
            lon_interval = 0.02
        elif lon_range < 0.5:
            lon_interval = 0.05
        else:
            lon_interval = 0.1

        # Generate tick locations based on ranges and intervals, rounding to the nearest multiple of the interval
        min_lat_rounded = np.floor(min_lat / lat_interval) * lat_interval
        min_lon_rounded = np.floor(min_lon / lon_interval) * lon_interval

        lat_ticks = np.arange(min_lat_rounded, max_lat + lat_interval, lat_interval)
        lon_ticks = np.arange(min_lon_rounded, max_lon + lon_interval, lon_interval)

        plt.yticks(lat_ticks)
        plt.xticks(lon_ticks)

        plt.xticks(fontsize=10)  # **Increased Tick Font Size**
        plt.yticks(fontsize=10)  # **Increased Tick Font Size**

        plt.tight_layout(h_pad=0.1)

        kmeans_filename = "kmeans.png"
        kmeans_filepath = os.path.join(app.config["PLOT_FOLDER"], kmeans_filename)
        plt.savefig(kmeans_filepath)
        plt.close()
    else:
        kmeans_filename = None

    cursor.close()
    conn.close()

    return (
        histogram_filename,
        heatmap_filename,
        kmeans_filename,
    )  # Returning the filenames


@app.route("/admin-page")
def admin_page():
    if "admin_id" not in session:
        return redirect(url_for("login"))

    histogram_filename, heatmap_filename, kmeans_filename = generate_and_save_plots()

    if not histogram_filename or not heatmap_filename:
        flash("No data available to display graphs.", "warning")
        histogram_url, heatmap_url, kmeans_url = None, None, None
    else:
        # Construct the URLs to the static files:
        histogram_url = url_for("static", filename=f"plots/{histogram_filename}")
        heatmap_url = url_for("static", filename=f"plots/{heatmap_filename}")
        kmeans_url = url_for("static", filename=f"plots/{kmeans_filename}")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch userform entries, handling potential None values, and joining with user table
    cursor.execute(
        """
        SELECT 
            uf.id, uf.name, u.phone_number, u.aadhaar_number, uf.issue_option, uf.`desc`, uf.area, uf.latitude, uf.longitude
        FROM 
            userforms uf
        LEFT JOIN 
            users u ON uf.user_id = u.id
        """
    )
    userform_entries = cursor.fetchall()

    # Ensure phone_number and aadhaar_number are not None before displaying
    userform_entries_safe = []
    for entry in userform_entries:
        safe_entry = list(entry)  # Convert tuple to list for modification
        safe_entry[2] = entry[2] if entry[2] is not None else "N/A"  # phone_number
        safe_entry[3] = entry[3] if entry[3] is not None else "N/A"  # aadhaar_number
        userform_entries_safe.append(safe_entry)

    cursor.close()
    conn.close()
    return render_template(
        "admin_page.html",
        histogram_url=histogram_url,  # URLs to the static plot files
        heatmap_url=heatmap_url,
        kmeans_url=kmeans_url,
        userform_entries=userform_entries_safe,  # From your existing code
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    session.pop("admin_id", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
