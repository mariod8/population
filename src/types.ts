export type PopulationData = {
  males: number[]
  females: number[]
}

export type LocationData = {
  [year: number]: PopulationData
}

export type Location = {
  code: string
  info: LocationData
}

export type Data = {
  [location: string]: Location
}
