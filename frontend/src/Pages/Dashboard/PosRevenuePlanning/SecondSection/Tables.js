import React, { Component } from "react";
import ChannelBudget from './ChannelBudget';
import CabinBudget from './CabinBudget';
import ODBudget from './ODBudget';
import '../PosRevenuePlanning.scss';

class Tables extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <ChannelBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <CabinBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            posDashboard={true}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <ODBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            {...this.props} />
        </div>
      </div >
    )
  }
}
export default Tables;