export type PopulationData = {
  males: number[]
  females: number[]
}

export type LocationData = {
  [year: number]: PopulationData
}

export type Location = {
  name: string
  info: LocationData
}

export type Data = {
  [location: string]: Location
}
