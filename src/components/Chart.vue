<template>
  <div class="select-none flex flex-col text-center w-full">
    <div class="my-4 flex flex-col">
      <span class="font-bold text-lg"
        >{{ data[location].name }} population in the year {{ year }}</span
      >
      <span class="font-bold text-lg">{{ numberWithCommas(countryPopulation) }}</span>
    </div>
    <div class="flex flex-col gap-[3px]">
      <div v-for="data in graphData" class="items-center flex flex-row [&>div]:leading-4">
        <div class="flex flex-row flex-1">
          <span class="mr-4">{{ formatNumber(data.males.amount) }}</span>
          <div
            :style="{
              width: `${data.males.ratio * 50}%`
            }"
            class="ml-auto h-4 min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2 scale-[-1]"
          ></div>
        </div>
        <div class="basis-[50px] max-w-[50px] text-center text-sm">
          {{ data.ageRange.start >= 100 ? "100+" : `${data.ageRange.start}-${data.ageRange.end}` }}
        </div>
        <div class="flex flex-row flex-1">
          <div
            :style="{
              width: `${data.males.ratio * 50}%`
            }"
            class="mr-auto h-4 min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2"
          ></div>
          <span class="ml-4">{{ formatNumber(data.females.amount) }}</span>
        </div>
      </div>
    </div>
    <div class="bg-white-2 bg-opacity-25 rounded-full h-1 w-full my-5"></div>
    <span class="font-bold text-lg">World population</span>
    <span class="font-bold text-2xl">{{ worldPopulation }}</span>
  </div>
</template>
<script lang="ts">
import type { Data, LocationData } from "@/types"
import { formatNumber, numberWithCommas } from "@/utils"
import type { PropType } from "vue"
import gsap from "gsap"

const ageGrouping = 5

export default {
  data() {
    return {
      worldPopulation: 0,
      tweenedWorldPopulation: 0
    }
  },
  methods: {
    formatNumber,
    numberWithCommas,
    population() {
      return [
        ...this.data[this.location].info[this.year].males,
        ...this.data[this.location].info[this.year].females
      ].reduce((sum, current) => sum + current, 0)
    },
    populationPerAge() {
      return this.data[this.location].info[this.year].males.map((value, i) => {
        return value + this.data[this.location].info[this.year].females[i]
      })
    },
    maxGenderValue() {
      return Math.max(
        ...[
          ...this.data[this.location].info[this.year].males,
          ...this.data[this.location].info[this.year].females
        ]
      )
    },
    percentage() {
      const population = this.population()

      return this.populationPerAge().map((value) => {
        const percent = value / population
        const product = percent < 0.001 ? 10000 : 1000
        const divisor = product === 10000 ? 100 : 10

        return Math.round(percent * product) / divisor
      })
    },
    calcWorldPopulation() {
      this.worldPopulation = [
        ...this.data["W"].info[this.year].males,
        ...this.data["W"].info[this.year].females
      ].reduce((sum, current) => sum + current, 0)
    },
    calcBarWidth(percent: number) {
      return percent
    }
  },
  computed: {
    countryPopulation() {
      return this.populationPerAge().reduce((sum, current) => sum + current, 0)
    },
    tweenWorldPopulation() {
      return gsap.to(this, {
        duration: 0.5,
        tweened: this.worldPopulation
      })
    },
    graphData() {
      const max = this.maxGenderValue()
      this.calcWorldPopulation()
      var age = 0

      return this.percentage().map((value, i) => {
        const data = {
          percentage: value,
          ageRange: { start: age, end: age + ageGrouping - 1 },
          males: {
            amount: this.data[this.location].info[this.year].males[i],
            ratio: this.data[this.location].info[this.year].males[i] / max
          },
          females: {
            amount: this.data[this.location].info[this.year].females[i],
            ratio: this.data[this.location].info[this.year].females[i] / max
          }
        }

        age += ageGrouping
        return data
      })
    }
  },
  props: {
    data: {
      type: Object as PropType<Data>,
      required: true
    },
    location: {
      type: String as PropType<keyof Data>,
      required: true
    },
    year: {
      type: Number as PropType<keyof LocationData>,
      required: true
    }
  }
}
</script>
