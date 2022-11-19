package com.ohero;

class OSMIssue {
    var imageId: Long = 0
    var latitude: Double = 0.0
    var longitude: Double = 0.0
    var osmWayId: Long = 0
    var isPrimary: Boolean = true
    var description: String = ""

    constructor(
        imageId: Long,
        latitude: Double,
        longitude: Double,
        osmWayId: Long,
        isPrimary: Boolean,
        description: String
    ) {
        this.imageId = imageId
        this.latitude = latitude
        this.longitude = longitude
        this.osmWayId = osmWayId
        this.isPrimary = isPrimary
        this.description = description
    }
}
