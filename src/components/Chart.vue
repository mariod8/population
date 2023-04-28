<template>
  <div class="flex flex-col w-[500px] bg-black-1 bg-opacity-75 p-5 rounded-3xl text-center">
    <span class="font-bold text-lg">{{ location }} population in {{ year }}</span>
    <div class="flex flex-col gap-[3px]">
      <div
        v-for="data in graphData"
        class="items-center flex flex-row [&>div]:leading-4 [&>div]:flex-auto"
      >
        <div class="">
          <div
            :style="{
              width: `${data.males.ratio * 100}%`,
              backgroundSize: `100% 100%`
            }"
            class="ml-auto h-4 min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2 scale-[-1] drop-shadow-md"
          ></div>
        </div>
        <div class="basis-[50px] max-w-[50px] text-center text-sm">
          {{ data.ageRange.start }}-{{ data.ageRange.end }}
        </div>
        <div class="">
          <div
            :style="{
              width: `${data.males.ratio * 100}%`,
              backgroundSize: `100% 100%`
            }"
            class="mr-auto h-4 min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>
<script lang="ts">
import type { Data, LocationData } from "@/types"
import { formatNumber } from "@/utils"
import type { PropType } from "vue"

const ageGrouping = 3

export default {
  data() {
    return {}
  },
  methods: {
    formatNumber,
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
    calcBarWidth(percent: number) {
      return percent
    }
  },
  computed: {
    graphData() {
      const max = this.maxGenderValue()
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
