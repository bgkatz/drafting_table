plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.bk.drawing"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.bk.drawing"
        minSdk = 33
        targetSdk = 34
        // CI passes -PciVersionCode=<run number> so each sideloaded
        // build installs as an update over the last one.
        versionCode = (project.findProperty("ciVersionCode") as String?)?.toIntOrNull() ?: 1
        versionName = "0.1"

        ndk {
            // MovinkPad 11 is arm64. Skip 32-bit/x86 to keep build fast.
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
            }
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        manifestPlaceholders["appLabel"] = "@string/app_name"
    }

    // Fixed key for the "ci" build type, so successive CI builds install
    // over each other. Debug-only throwaway key (password "android",
    // same as Android's own debug keystore); never use it for release.
    signingConfigs {
        create("ci") {
            storeFile = file("ci-debug.keystore")
            storePassword = "android"
            keyAlias = "cidebug"
            keyPassword = "android"
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        // Sideload build from GitHub Actions (.github/workflows/ci-apk.yml).
        // Separate application id + label, so it installs NEXT TO the
        // USB-deployed debug app instead of replacing it. Both share the
        // /sdcard/Drafting Table documents folder — don't run both at once.
        create("ci") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".ci"
            versionNameSuffix = "-ci"
            signingConfig = signingConfigs.getByName("ci")
            manifestPlaceholders["appLabel"] = "Drafting Table CI"
            matchingFallbacks += listOf("debug")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.0")

    // Low-latency front-buffered GL renderer (the crown jewel).
    implementation("androidx.graphics:graphics-core:1.0.0-alpha05")

    // Stylus input prediction — used in Spike 2.
    implementation("androidx.input:input-motionprediction:1.0.0-beta04")

    // Instrumented (on-device) test harness for renderer fidelity tests.
    // Runs against the connected tablet with a private EGL pbuffer
    // context — no UI, no SurfaceView. See app/src/androidTest/.
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
}
