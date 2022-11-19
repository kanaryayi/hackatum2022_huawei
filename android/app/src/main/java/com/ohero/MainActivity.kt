package com.ohero

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.*
import common.helpers.InstantPlacementSettings
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker


class MainActivity : AppCompatActivity() {
    private var mMapView: MapView? = null
    private var mMapController: MapController? = null

    lateinit var arCoreSessionHelper: ARCoreSessionLifecycleHelper
    lateinit var view: HelloArView
    lateinit var renderer: HelloArRenderer

    val instantPlacementSettings = InstantPlacementSettings()

    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestPermissions(
            arrayOf(
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.INTERNET,
                Manifest.permission.MANAGE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA
            ), 100
        )

        Configuration.getInstance().setUserAgentValue(packageName)
        mMapView = findViewById<View>(R.id.map_view) as MapView
        mMapView!!.setTileSource(TileSourceFactory.DEFAULT_TILE_SOURCE)
        mMapView!!.setBuiltInZoomControls(true)
        mMapController = mMapView!!.getController() as MapController
        mMapController!!.setZoom(13)
        val gPt = GeoPoint(48.26266548288217, 11.667834830722132)
        mMapController!!.setCenter(gPt)
        mMapController!!.setZoom(18)

        readMarkers(mMapView!!)

        // Setup ARCore session lifecycle helper and configuration.
        arCoreSessionHelper = ARCoreSessionLifecycleHelper(this)
        // If Session creation or Session.resume() fails, display a message and log detailed
        // information.
        arCoreSessionHelper.exceptionCallback =
            { exception ->
                val message =
                    when (exception) {
                        is UnavailableUserDeclinedInstallationException ->
                            "Please install Google Play Services for AR"
                        is UnavailableApkTooOldException -> "Please update ARCore"
                        is UnavailableSdkTooOldException -> "Please update this app"
                        is UnavailableDeviceNotCompatibleException -> "This device does not support AR"
                        is CameraNotAvailableException -> "Camera not available. Try restarting the app."
                        else -> "Failed to create AR session: $exception"
                    }
                Log.e("TAG", "ARCore threw an exception", exception)
                view.snackbarHelper.showError(this, message)
            }

        // Configure session features, including: Lighting Estimation, Depth mode, Instant Placement.
        arCoreSessionHelper.beforeSessionResume = ::configureSession
        lifecycle.addObserver(arCoreSessionHelper)

        val okButton = findViewById<View>(R.id.info_ok_button) as Button
        val infoView = findViewById<View>(R.id.info_view) as ConstraintLayout
        if (intent.hasExtra("showInfo")) {
            if (intent.getBooleanExtra("showInfo", false)) {
                infoView.visibility = View.VISIBLE
            } else {
                val goldCoinText = findViewById<View>(R.id.coin_text) as TextView
                goldCoinText.text = "1099"
            }
        } else {
            infoView.visibility = View.VISIBLE
        }
        okButton.setOnClickListener {
            infoView.visibility = View.GONE
        }
    }

    // Configure the session, using Lighting Estimation, and Depth mode.
    fun configureSession(session: Session) {
        session.configure(
            session.config.apply {
                lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR

                // Depth API is used if it is configured in Hello AR's settings.
                depthMode =
                    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                        Config.DepthMode.AUTOMATIC
                    } else {
                        Config.DepthMode.DISABLED
                    }

                // Instant Placement is used if it is configured in Hello AR's settings.
                instantPlacementMode =
                    if (instantPlacementSettings.isInstantPlacementEnabled) {
                        Config.InstantPlacementMode.LOCAL_Y_UP
                    } else {
                        Config.InstantPlacementMode.DISABLED
                    }
            }
        )
    }

    private fun addMarker(osmIssue: OSMIssue, mapView: MapView) {
        val startPoint: GeoPoint = GeoPoint(osmIssue.latitude, osmIssue.longitude)
        val marker = Marker(mapView)
        marker.setIcon(
            if (osmIssue.imageId == 123L) {
                resources.getDrawable(R.drawable.chest)
            } else {
                resources.getDrawable(R.drawable.chest_gray)
            }
        )
        marker.position = startPoint
        marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)

        if (osmIssue.imageId == 123L) {
            marker.setOnMarkerClickListener { marker2, mapView2 ->
                startActivity(
                    Intent(this, ARActivity::class.java).apply {})
                true
            }
        }
        mapView.getOverlays().add(marker)
    }

    private fun readMarkers(mapView: MapView) {
        val issueList: List<OSMIssue> = listOf(
            OSMIssue(
                123,
                48.26266548288217,
                11.667834830722132,
                123,
                false,
                "Test"
            ),
            OSMIssue(
                173529288004582,
                48.111355593621,
                11.61430161647961,
                154359841,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                187098909929555,
                48.17566584451249,
                11.606248275586244,
                38690109,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                216283226982432,
                48.13569262741428,
                11.63329945842623,
                460354029,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                269290068323331,
                48.1765304782419,
                11.58813361092573,
                259583217,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                305399287917735,
                48.14912076050485,
                11.570784175202212,
                122159109,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                308707770748550,
                48.17607553069282,
                11.60678544203654,
                38690109,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                310698823750341,
                48.106241100000005,
                11.5806306,
                362796882,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                315970676772352,
                48.13046701048668,
                11.613351477617488,
                122002285,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                334213504878448,
                48.16163430223077,
                11.622674470519335,
                38926857,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                387616345626796,
                48.172420882348,
                11.608645678780396,
                99663070,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                432081875138552,
                48.11291668736461,
                11.547805204701106,
                173096616,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                486527229463839,
                48.16265466072837,
                11.6237920164228,
                100945239,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                497792407930066,
                48.118031503459626,
                11.61900312193212,
                49915865,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                507005454009469,
                48.17604819685549,
                11.60673962645799,
                38690109,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                518223702859508,
                48.12177302648248,
                11.62046123985182,
                292781203,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                532051091386866,
                48.11290549141834,
                11.615932367309346,
                32256065,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                568493717757567,
                48.11925483880852,
                11.61818097587384,
                27334574,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                677806630089122,
                48.16881681797188,
                11.616833182061308,
                38123963,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                742794133052728,
                48.104692400000005,
                11.5873091,
                40108148,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                796584247628919,
                48.17208671297508,
                11.603695515414293,
                38690109,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                811604529752448,
                48.11211300211448,
                11.615292419261202,
                37666015,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                821061125222211,
                48.14868644684231,
                11.525206316022905,
                54200279,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                824423578448957,
                48.171796512604,
                11.601411486290576,
                159686837,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                827676088105008,
                48.16089514763812,
                11.549527023953246,
                90165750,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                877953149735095,
                48.11835266871433,
                11.618987385257748,
                230251455,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                927816371125415,
                48.16302274431093,
                11.624135105357796,
                467655365,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                952367918873603,
                48.10089803370931,
                11.541217267708962,
                262486709,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                1998769373606723,
                48.11503273780353,
                11.617648147817643,
                311181487,
                false,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                2096254440714769,
                48.17299008497127,
                11.613106928911195,
                37208105,
                true,
                "Solve the riddle of missing jewels."
            ),
            OSMIssue(
                2947015848899610,
                48.14774795111176,
                11.527965543942017,
                54200280,
                false,
                "Solve the riddle of missing jewels."
            ),
        );

        for (osmIssue in issueList) {
            addMarker(osmIssue, mapView)
        }
    }
}